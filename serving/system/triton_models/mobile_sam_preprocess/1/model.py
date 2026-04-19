"""
Triton Python backend — CPU preprocessing stage.

Inputs:
  INPUT_IMAGE  TYPE_STRING [1]  base64 JPEG
  BBOX         TYPE_FP32   [4]  [x, y, w, h]  (all -1 = centre point fallback)

Outputs:
  IMAGE_TENSOR  TYPE_FP32 [1,3,1024,1024]  normalised image ready for encoder
  POINT_COORDS  TYPE_FP32 [1,5,2]          scaled 1024-space coordinates
  POINT_LABELS  TYPE_FP32 [1,5]
  ORIG_SIZE     TYPE_FP32 [2]              [orig_h, orig_w] for decoder
"""
import base64, io
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

PIXEL_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
PIXEL_STD  = np.array([ 58.395,  57.12,  57.375], dtype=np.float32)


def _preprocess(image_rgb: np.ndarray, size: int = 1024) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
    resized = np.array(Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR), dtype=np.float32)
    padded = np.zeros((size, size, 3), dtype=np.float32)
    padded[:new_h, :new_w] = resized
    return ((padded - PIXEL_MEAN) / PIXEL_STD).transpose(2, 0, 1)[None]


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            img_b64 = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE").as_numpy()[0]
            if isinstance(img_b64, bytes):
                img_b64 = img_b64.decode()
            image = np.array(Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB"))
            orig_h, orig_w = image.shape[:2]
            bbox = pb_utils.get_input_tensor_by_name(request, "BBOX").as_numpy().flatten().tolist()

            image_tensor = _preprocess(image)  # (1, 3, 1024, 1024)

            scale = 1024 / max(orig_h, orig_w)
            if all(v >= 0 for v in bbox):
                x, y, w, h = [c * scale for c in bbox]
                x1, y1, x2, y2 = x, y, x + w, y + h
                point_coords = np.array([[[x1,y1],[x2,y2],[0,0],[0,0],[0,0]]], dtype=np.float32)
                point_labels = np.array([[2, 3, -1, -1, -1]],                  dtype=np.float32)
            else:
                point_coords = np.array([[[orig_w/2*scale, orig_h/2*scale],[0,0],[0,0],[0,0],[0,0]]], dtype=np.float32)
                point_labels = np.array([[1, -1, -1, -1, -1]],                                        dtype=np.float32)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("IMAGE_TENSOR", image_tensor),
                pb_utils.Tensor("POINT_COORDS", point_coords),
                pb_utils.Tensor("POINT_LABELS", point_labels),
                pb_utils.Tensor("ORIG_SIZE",    np.array([orig_h, orig_w], dtype=np.float32)),
            ]))
        return responses
