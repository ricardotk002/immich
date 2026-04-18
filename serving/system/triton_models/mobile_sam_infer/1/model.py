"""
Triton Python backend — GPU inference stage.

Inputs:
  IMAGE_TENSOR  TYPE_FP32 [1,3,1024,1024]
  POINT_COORDS  TYPE_FP32 [1,5,2]
  POINT_LABELS  TYPE_FP32 [1,5]
  ORIG_SIZE     TYPE_FP32 [2]              [orig_h, orig_w]

Outputs:
  MASK         TYPE_STRING [1]  base64 PNG
  ENCODER_MS   TYPE_FP32   [1]
  DECODER_MS   TYPE_FP32   [1]
"""
import base64, io, os, time
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class TritonPythonModel:
    def initialize(self, args):
        import onnxruntime as ort
        device_id = int(args.get("model_instance_device_id", 0))
        data_dir  = os.environ.get("MODEL_DIR", "/data")
        opts = [{"device_id": device_id, "gpu_mem_limit": 2 * 1024 * 1024 * 1024}, {}]
        self.enc_sess = ort.InferenceSession(
            os.path.join(data_dir, "mobile_sam_encoder.onnx"),
            providers=PROVIDERS, provider_options=opts)
        self.dec_sess = ort.InferenceSession(
            os.path.join(data_dir, "mobile_sam_decoder.onnx"),
            providers=PROVIDERS, provider_options=opts)

    def execute(self, requests):
        responses = []
        for request in requests:
            image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_TENSOR").as_numpy()
            point_coords = pb_utils.get_input_tensor_by_name(request, "POINT_COORDS").as_numpy()
            point_labels = pb_utils.get_input_tensor_by_name(request, "POINT_LABELS").as_numpy()
            orig_size    = pb_utils.get_input_tensor_by_name(request, "ORIG_SIZE").as_numpy()
            orig_h, orig_w = int(orig_size[0]), int(orig_size[1])

            t0 = time.perf_counter()
            (embed,) = self.enc_sess.run(["image_embeddings"], {"image": image_tensor})
            enc_ms = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            masks, _, _ = self.dec_sess.run(
                ["masks", "iou_predictions", "low_res_masks"],
                {
                    "image_embeddings": embed,
                    "point_coords":     point_coords,
                    "point_labels":     point_labels,
                    "mask_input":       np.zeros((1, 1, 256, 256), dtype=np.float32),
                    "has_mask_input":   np.array([0],              dtype=np.float32),
                    "orig_im_size":     np.array([orig_h, orig_w], dtype=np.float32),
                },
            )
            dec_ms = (time.perf_counter() - t0) * 1e3

            buf = io.BytesIO()
            Image.fromarray((masks[0, 0] > 0).astype(np.uint8) * 255).save(buf, format="PNG")
            mask_b64 = base64.b64encode(buf.getvalue()).decode()

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("MASK",       np.array([mask_b64],  dtype=object)),
                pb_utils.Tensor("ENCODER_MS", np.array([enc_ms],    dtype=np.float32)),
                pb_utils.Tensor("DECODER_MS", np.array([dec_ms],    dtype=np.float32)),
            ]))
        return responses
