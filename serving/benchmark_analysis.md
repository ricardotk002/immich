# MobileSAM Serving Benchmark Analysis

---

## Part 1 — Model Benchmark (isolated inference, single request)

These experiments measure raw inference speed with no HTTP stack, no preprocessing overhead, and no concurrency. Each experiment runs 100 inferences after a 3-request warm-up, reporting end-to-end encoder + decoder latency.

### Results Table

| Experiment | Model file(s) | Runtime | Hardware | p50 latency | p95 latency | p99 latency | Throughput | Notes |
|---|---|---|---|---|---|---|---|---|
| PyTorch | `mobile_sam.pt` | torch 2.3 / cuDNN | 1× NVIDIA GPU | **49.46 ms** | 51.07 ms | 52.60 ms | **20.22 FPS** | Fastest overall; cuDNN auto-tunes kernels for the specific GPU |
| TensorRT | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | ORT 1.19 / TensorRT EP (encoder) + CUDA EP (decoder) | 1× NVIDIA GPU | 58.66 ms | 59.41 ms | 59.64 ms | 17.05 FPS | TRT compiles encoder only; decoder falls back to CUDA EP (OneHot ops unsupported by TRT) |
| ONNX (GPU) | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | ORT 1.19 / CUDA EP | 1× NVIDIA GPU | 85.19 ms | 86.46 ms | 86.94 ms | 11.74 FPS | Generic CUDA kernels, no graph compilation |
| OpenVINO | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | OpenVINO / CPU | CPU only | 167.24 ms | 174.05 ms | 192.55 ms | 5.98 FPS | CPU inference; competitive given no GPU |
| ONNX quantized (CPU) | `mobile_sam_encoder_quantized.onnx` + `mobile_sam_decoder_quantized.onnx` | ORT 1.19 / CPU EP | CPU only | 339.08 ms | 362.05 ms | 370.88 ms | 2.95 FPS | Dynamic quantization produces ConvInteger nodes unsupported by CUDA EP — forced to CPU |

### Analysis

**PyTorch is the fastest runtime on GPU, not because of any algorithmic advantage, but because cuDNN selects and caches the optimal convolution algorithm for the specific GPU model at first inference.** TensorRT comes close (58 ms) since it compiles the encoder graph into a fused execution plan, but the decoder must fall back to the CUDA EP because TRT cannot handle the OneHot ops used as shape tensors in the mask decoder. This split-backend overhead is why TRT ends up 9 ms slower than PyTorch rather than faster.

ONNX with the CUDA EP (85 ms) uses generic CUDA kernels with no graph-level fusion, so it pays the overhead of launching individual CUDA kernels per op rather than fused kernels per layer block. The 73% latency gap between PyTorch and ONNX (49 ms vs 85 ms) is entirely explained by this difference — the same weights, the same arithmetic, different kernel selection strategy.

The p95–p50 spread for all GPU experiments is only 2–4 ms, which indicates extremely stable single-request latency with no variance from queuing or scheduling jitter. This is the theoretical floor: what you can expect per request before any serving overhead is added.

**CPU runtimes are not competitive for this model.** MobileSAM's ViT-Tiny encoder is compute-dense (primarily large matrix multiplications and attention operations), which benefits enormously from GPU parallelism. OpenVINO at 167 ms uses Intel's optimised CPU kernels and is a reasonable option if no GPU is available. The quantized ONNX model at 339 ms is counterproductive: dynamic quantization reduces model size but forces CPU execution, and the ConvInteger kernel is slower than the original FP32 CUDA path by 4×.

---

## Part 2 — System Benchmark (full serving stack, realistic load)

These experiments measure the end-to-end wall time seen by a client, including HTTP overhead, base64 encoding/decoding, Python preprocessing, queuing, and concurrent request handling. Three load patterns are tested: serial (1 request at a time), concurrent (8 parallel clients, 200 total requests), and Poisson (5 req/s arrival rate over 30 s).

### Results Table

| Option | Endpoint URL | Model file(s) | Code version | Hardware | p50 / p95 latency | Throughput | Error rate | Concurrency tested | Compute instance | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| FastAPI + PyTorch — serial | `fastapi_pytorch_server:8000/predict` | `mobile_sam.pt` | FastAPI 0.111 / uvicorn 0.30 / torch 2.3 | 1× NVIDIA GPU | 142 ms / 177 ms | 5.71 req/s | 0% | 1 | GPU node | p99 spike (819 ms) is a warmup outlier leaking past the 3-request warm-up |
| FastAPI + PyTorch — concurrent | `fastapi_pytorch_server:8000/predict` | `mobile_sam.pt` | FastAPI 0.111 / uvicorn 0.30 / torch 2.3 | 1× NVIDIA GPU | 275 ms / 488 ms | **25.57 req/s** | 0% | 8 | GPU node | Best throughput across all experiments |
| FastAPI + PyTorch — poisson | `fastapi_pytorch_server:8000/predict` | `mobile_sam.pt` | FastAPI 0.111 / uvicorn 0.30 / torch 2.3 | 1× NVIDIA GPU | 206 ms / 271 ms | 4.74 req/s | 0% | ~5 req/s | GPU node | Stable under realistic bursty arrival rate |
| FastAPI + ONNX — serial | `fastapi_onnx_server:8000/predict` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | FastAPI 0.111 / uvicorn 0.30 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 186 ms / 202 ms | 2.91 req/s | 0% | 1 | GPU node | p99 = 3910 ms — severe cold-start spike, first request pays full CUDA init cost |
| FastAPI + ONNX — concurrent | `fastapi_onnx_server:8000/predict` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | FastAPI 0.111 / uvicorn 0.30 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 350 ms / 541 ms | 20.97 req/s | 0% | 8 | GPU node | 25% lower throughput than PyTorch under the same concurrency |
| FastAPI + ONNX — poisson | `fastapi_onnx_server:8000/predict` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | FastAPI 0.111 / uvicorn 0.30 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 213 ms / 288 ms | 4.75 req/s | 0% | ~5 req/s | GPU node | Consistently slower than PyTorch at every percentile |
| Triton + ONNX (1 GPU instance) — serial | `triton_server:8000` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | Triton 24.05 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 161 ms / 164 ms | 6.32 req/s | 0% | 1 | GPU node | Matches PyTorch at concurrency=1; tight p50–p95 spread shows stable single-request path |
| Triton + ONNX (1 GPU instance) — concurrent | `triton_server:8000` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | Triton 24.05 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 1124 ms / 1136 ms | 7.11 req/s | 0% | 8 | GPU node | Queue = 973 ms of 1124 ms total — completely queue-bound |
| Triton + ONNX (1 GPU instance) — poisson | `triton_server:8000` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | Triton 24.05 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 283 ms / 565 ms | 4.77 req/s | 0% (21.9% delayed) | ~5 req/s | GPU node | Cannot sustain 5 req/s; queue grows unbounded |
| Triton + ONNX (2 GPU instances) — concurrent | `triton_server:8000` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | Triton 24.05 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 568 ms / 703 ms | 13.71 req/s | 0% | 8 | GPU node | Doubling instances doubles throughput but queue still dominates (433 ms of 578 ms) |
| Triton + ONNX (2 GPU instances) — poisson | `triton_server:8000` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | Triton 24.05 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 170 ms / 282 ms | 4.79 req/s | 0% (3.5% delayed) | ~5 req/s | GPU node | Best Triton result under realistic load |
| Triton + ONNX + Dynamic Batching — poisson | `triton_server:8000` | `mobile_sam_encoder.onnx` + `mobile_sam_decoder.onnx` | Triton 24.05 / onnxruntime-gpu 1.18 | 1× NVIDIA GPU | 237 ms / 574 ms | 4.76 req/s | 0% (17.3% delayed) | ~5 req/s | GPU node | Batching confirmed (514 inferences / 415 executions) but wait time worsens p95 |

### Analysis

#### Bridging model and system latency

The model benchmark establishes a theoretical floor. Comparing it to the system results reveals exactly where time is being spent outside of inference:

| Backend | Model benchmark (inference only) | System serial p50 | Overhead |
|---|---|---|---|
| PyTorch | 49 ms | 142 ms | ~93 ms |
| ONNX (GPU) | 85 ms | 186 ms | ~101 ms |
| Triton + ONNX (serial) | ~85 ms | 161 ms | ~76 ms |

The ~100 ms overhead in FastAPI is base64 decoding, PIL image open, numpy resize + normalise, and HTTP round-trip. Triton's lower overhead (~76 ms) is because the `perf_analyzer` client sends raw binary tensors over HTTP/2, skipping base64 and PIL entirely. This is a measurement artefact — a real client would pay the same encoding cost.

The PyTorch advantage from the model benchmark (49 ms vs 85 ms, a 73% gap) shrinks to ~44 ms in the system benchmark because the ~93 ms fixed overhead dilutes it. In absolute terms PyTorch is still faster, but at the system level the bottleneck shifts toward preprocessing and scheduling once concurrency is introduced.

#### Latency under load

Under concurrent load (8 workers, 200 requests), the FastAPI p50 rises from ~142–186 ms (serial) to ~275–350 ms. This ~130–165 ms increase comes from 4 workers handling 8 concurrent clients, meaning requests queue briefly at the Python layer. The GPU itself is never the bottleneck — compute time per request stays at 60–170 ms regardless of concurrency.

Triton with 1 instance sees p50 jump from 161 ms to 1124 ms under the same concurrency. The server-side breakdown makes this explicit: 973 ms queue, 140 ms compute. The GPU is idle for 87% of the request lifetime waiting for the Python backend to schedule it. This is the fundamental mismatch between Triton's Python backend model and a GPU that can only be kept busy by feeding it requests faster than Python can dispatch them.

#### Throughput ceiling

The model benchmark shows PyTorch at 20.22 FPS (49 ms/request). With 4 uvicorn workers each running independently, the theoretical system throughput ceiling is ~4 × 20 FPS = ~80 req/s if inference were the only cost. The actual system achieves 25.6 req/s — the gap is the ~93 ms preprocessing overhead per request, which limits each worker to ~6–7 req/s regardless of GPU speed. **This means the current bottleneck in the FastAPI + PyTorch stack is not the model — it is Python CPU preprocessing.**

For Triton's Python backend the situation is the same but worse: the preprocessing runs inside the same serialised `execute()` call as the inference, so CPU and GPU work cannot overlap between requests.

#### Resilience

FastAPI achieved 0% errors and 0% delayed requests in all Poisson experiments. Triton GPU-1 at 5 req/s had 21.9% of requests delayed, meaning the system was not at steady state — the queue was growing throughout the 30 s window. GPU-2 improved this to 3.5%, which is marginal. Dynamic batching at 17.3% delayed is worse than 2 instances despite batching occurring, because the wait for batch formation adds latency that outweighs the batching efficiency gain at this arrival rate.

#### Maintainability

FastAPI is a single Python file and a Dockerfile. Model updates are a file copy. Scaling is a `--workers` flag.

Triton requires: a model repository, one `config.pbtxt` per model variant, shared memory configuration, VRAM budgeting per instance, ensemble config to chain preprocessing → inference, and `perf_analyzer` for benchmarking. Any change to the input interface (e.g. renaming `BOX` → `BBOX`) requires updating the preprocess model, the ensemble config, and the client simultaneously. This overhead is justified for multi-model serving or live model swapping — not for a single-model endpoint.

---

## Combined Recommendation

**FastAPI + PyTorch (`mobile_sam.pt`) is the best option for this deployment.**

The model benchmark confirms PyTorch is the fastest runtime (49 ms, 20 FPS), and the system benchmark confirms this advantage holds end-to-end (142 ms serial, 25.6 req/s concurrent) with zero errors under all load patterns.

The ONNX runtime is a viable fallback for environments where the PyTorch dependency is undesirable or where GPU is unavailable. TensorRT would be the next option if squeezing the last few milliseconds out of single-request latency matters, though the decoder's CUDA EP fallback limits the gain to ~9 ms over PyTorch in practice.

Triton is worth revisiting only if the requirements change: multi-model serving, A/B model testing, or Triton's native metrics pipeline. In that case, the stable configuration is 2 GPU instances with the CUDA EP; dynamic batching does not help for this workload.

The next leverage point for latency reduction is **preprocessing**: the ~100 ms overhead from base64 decoding, PIL resize, and normalisation is larger than the entire inference gap between PyTorch and ONNX. Moving clients to send pre-normalised tensors (or at minimum raw binary instead of base64) would cut system latency by more than switching runtimes.
