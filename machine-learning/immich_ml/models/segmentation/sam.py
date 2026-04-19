from __future__ import annotations

import base64
import io
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from immich_ml.config import log
from immich_ml.models.base import InferenceModel
from immich_ml.schemas import ModelTask, ModelType

PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

MOBILESAM_CHECKPOINT_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt"
)


def _preprocess(image_rgb: NDArray[np.uint8], size: int = 1024) -> NDArray[np.float32]:
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = np.array(Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR), dtype=np.float32)
    padded = np.zeros((size, size, 3), dtype=np.float32)
    padded[:new_h, :new_w] = resized
    return ((padded - PIXEL_MEAN) / PIXEL_STD).transpose(2, 0, 1)[None]


class SAMSegmenter(InferenceModel):
    depends = []
    identity = (ModelType.PIPELINE, ModelTask.SEGMENTATION)

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self.bbox: list[float] | None = None
        self.point_coords: list[list[float]] | None = None
        self.decoder_session: Any = None

    @property
    def checkpoint_path(self) -> Path:
        return self.cache_dir / "mobile_sam.pt"

    @property
    def encoder_path(self) -> Path:
        return self.cache_dir / "mobile_sam_encoder.onnx"

    @property
    def decoder_path(self) -> Path:
        return self.cache_dir / "mobile_sam_decoder.onnx"

    @property
    def cached(self) -> bool:
        return self.encoder_path.is_file() and self.decoder_path.is_file()

    def _download(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.checkpoint_path.exists():
            log.info(f"Downloading MobileSAM checkpoint to {self.checkpoint_path}")
            urllib.request.urlretrieve(MOBILESAM_CHECKPOINT_URL, self.checkpoint_path)
            log.info("MobileSAM checkpoint downloaded.")

        if not (self.encoder_path.exists() and self.decoder_path.exists()):
            self._export_onnx()

    def _export_onnx(self) -> None:
        try:
            import torch
            from mobile_sam import sam_model_registry
            from mobile_sam.utils.onnx import SamOnnxModel
        except ImportError as e:
            raise RuntimeError(
                "torch and mobile-sam are required to export MobileSAM to ONNX. "
                "Install with: pip install torch mobile-sam"
            ) from e

        log.info("Exporting MobileSAM to ONNX (this may take a few minutes)...")
        sam = sam_model_registry["vit_t"](checkpoint=str(self.checkpoint_path))
        sam.eval()

        if not self.encoder_path.exists():
            log.info(f"Exporting encoder → {self.encoder_path}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    sam.image_encoder,
                    torch.randn(1, 3, 1024, 1024),
                    str(self.encoder_path),
                    input_names=["image"],
                    output_names=["image_embeddings"],
                    opset_version=17,
                )

        if not self.decoder_path.exists():
            log.info(f"Exporting decoder → {self.decoder_path}")
            onnx_model = SamOnnxModel(sam, return_single_mask=True).eval()
            embed_dim = sam.prompt_encoder.embed_dim
            embed_size = sam.prompt_encoder.image_embedding_size
            dummy = (
                torch.randn(1, embed_dim, *embed_size),
                torch.randint(0, 1024, (1, 5, 2), dtype=torch.float32),
                torch.randint(0, 4, (1, 5), dtype=torch.float32),
                torch.randn(1, 1, 256, 256),
                torch.tensor([1], dtype=torch.float32),
                torch.tensor([1024, 1024], dtype=torch.float32),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    onnx_model,
                    dummy,
                    str(self.decoder_path),
                    input_names=[
                        "image_embeddings",
                        "point_coords",
                        "point_labels",
                        "mask_input",
                        "has_mask_input",
                        "orig_im_size",
                    ],
                    output_names=["masks", "iou_predictions", "low_res_masks"],
                    dynamic_axes={"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}},
                    opset_version=17,
                )

        log.info("MobileSAM ONNX export complete.")

    def _load(self) -> Any:
        from immich_ml.sessions.ort import OrtSession

        self.decoder_session = OrtSession(self.decoder_path)
        return OrtSession(self.encoder_path)

    def configure(self, **kwargs: Any) -> None:
        self.bbox = kwargs.get("bbox")
        self.point_coords = kwargs.get("point_coords")

    def _predict(self, image: Image.Image) -> dict[str, Any]:
        if self.bbox is None and self.point_coords is None:
            raise ValueError("Either 'bbox' or 'point_coords' must be provided in options")

        image_np = np.array(image.convert("RGB"))
        orig_h, orig_w = image_np.shape[:2]
        scale = 1024 / max(orig_h, orig_w)

        t0 = time.perf_counter()
        (embedding,) = self.session.run(["image_embeddings"], {"image": _preprocess(image_np)})
        encoder_ms = (time.perf_counter() - t0) * 1e3

        if self.bbox is not None:
            x, y, w, h = [c * scale for c in self.bbox]
            x1, y1, x2, y2 = x, y, x + w, y + h
            point_coords = np.array([[[x1, y1], [x2, y2], [0, 0], [0, 0], [0, 0]]], dtype=np.float32)
            point_labels = np.array([[2, 3, -1, -1, -1]], dtype=np.float32)
        else:
            px = self.point_coords[0][0] * scale  # type: ignore[index]
            py = self.point_coords[0][1] * scale  # type: ignore[index]
            point_coords = np.array([[[px, py], [0, 0], [0, 0], [0, 0], [0, 0]]], dtype=np.float32)
            point_labels = np.array([[1, -1, -1, -1, -1]], dtype=np.float32)

        t0 = time.perf_counter()
        masks, iou, _ = self.decoder_session.run(
            ["masks", "iou_predictions", "low_res_masks"],
            {
                "image_embeddings": embedding,
                "point_coords": point_coords,
                "point_labels": point_labels,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.array([0], dtype=np.float32),
                "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32),
            },
        )
        decoder_ms = (time.perf_counter() - t0) * 1e3

        mask = (masks[0, 0] > 0).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(mask).save(buf, format="PNG")

        return {
            "mask": base64.b64encode(buf.getvalue()).decode(),
            "iou_score": float(iou[0, 0]),
            "encoder_ms": encoder_ms,
            "decoder_ms": decoder_ms,
        }
