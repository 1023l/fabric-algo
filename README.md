[English](README.md) | [简体中文](README.zh-CN.md)

# fabric-algo

A general-purpose visual detection inference engine: **object detection + OCR recognition + real-time counting + third-party HTTP API**.

> The project is a detection & recognition pipeline decoupled from any specific industry (detect objects → recognize text regions → track & count).
> Currently shipped with a **textile fabric counting / shoe-size recognition** reference implementation. The repo retains the `fabric-algo` name;
> model/dataset identifiers (`fabric` / `text` / `rec`) are legacy dataset codes, not industry-bound.

Works with the annotation & training platform [train-center](https://github.com/1023l/train-center): it produces YOLO `.pt` and PaddleOCR inference models; this repo converts them to TensorRT engines and serves them online.

## Features

- Web demo: upload video, drag counting line, forward/reverse crossing count, object attributes (L/R, pairing, recognized text)
- Algorithm API: `POST /api/algo/process` (third-party software calls once per frame)
- Inference backend: TensorRT 11 engine (primary path) or ONNX Runtime

See [docs/算法接口文档.md](./docs/算法接口文档.md) for API details.

## Quick Start

```bash
# Recommended: conda env yolo-bench (see requirements-yolo-bench.txt)
python web_server.py --host 0.0.0.0 --port 8001
```

- UI: http://127.0.0.1:8001/
- Swagger: http://127.0.0.1:8001/docs
- Health check: http://127.0.0.1:8001/health

Default model loading (place weights in the corresponding directories; weights are not committed):

```
models/fabric/fabric20260828V2.engine
models/text/text20260831V1.engine
models/rec/rec20260828V3.engine
```

The `inference.yml` (character table) must sit next to the rec engine; decoding depends on it.

## From train-center to This Repo

Use the unified `tools/` scripts (configuration centralized in [tools/config.yaml](./tools/config.yaml) — change only this file when switching machines or releasing new versions):

```bash
# 1. Export ONNX (det uses yolo-bench env; rec auto-switches to paddle-ocr env)
python tools/export_onnx.py det --model all --version 20260828V2
python tools/export_onnx.py rec --version 20260828V3

# 2. Build TRT engines (trt_cache timing cache accelerates rebuilds; logs/intermediates in trt_export)
python tools/build_engine.py --model all

# 3. Verify & benchmark
python tools/verify.py engine          # load + infer all 3 engines
python tools/verify.py onnx --model all
python tools/bench.py --model fabric

# 4. rec accuracy (per-sample prediction vs. label)
python tools/eval_rec.py --backend onnx

# 5. Data / API utilities
python tools/crop_pieces.py --video xx.mp4 --out data_pieces/   # source for text annotation
python tools/restore_det_images.py --dataset det_text           # det data recovery
python tools/api_smoke.py                                       # API smoke test
```

Runtime does not require PyTorch / PaddlePaddle; only the `tools/` export/build step does (env auto-switches).

## Model Management (Upload → Online)

The "Model Management" panel at the bottom of the page supports two upload modes:

```
POST /api/models/upload         # Bundle: model_bundle_*.zip (train-center "one-click export 3 models" product)
POST /api/models/upload_single  # Single model: fabric/text .pt; rec rec*.zip or model folder
GET  /api/models/task           # Conversion progress (export ONNX -> TRT engine -> hot-swap)
GET  /api/models/current        # Current active version (models/current.json)
```

- **Bundle**: replaces all 3 models at once; **Single model**: only converts the uploaded one, others stay at current version
- Single-model naming convention: `fabric20260910V1.pt` / `text20260910V1.pt` / rec folder `rec20260910V1/` (must contain best.pdparams)
- Background auto-completes ONNX export, TRT engine build, and **trial-loads before hot-swapping** the online model (on failure, the old model keeps serving)
- Uploaded version must differ from the current active version (dedup; rebuild with a new version number if needed)

## Docker Deployment

Containerized deployment config lives in [deploy/](./deploy/), based on `nvidia/cuda:12.4.1-runtime-ubuntu24.04` image with TRT, torch, ultralytics and other dependencies.

```bash
cd fabric-algo
docker compose -f deploy/docker-compose.yml up -d --build
```

> Requires NVIDIA driver + nvidia-container-toolkit on the target machine (native Linux or Windows WSL2). First build downloads large dependencies; subsequent builds use cache.

## Dependencies

- Lightweight ONNX: `requirements.txt`
- On-site GPU + TRT: `requirements-yolo-bench.txt` (includes `tensorrt_cu12==11.2`, cupy)
- Docker container: `deploy/requirements-docker.txt`
