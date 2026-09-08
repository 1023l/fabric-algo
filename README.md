# fabric-algo

鞋型布柔性材料字符视觉检测：**实时计数 + OCR + 第三方 HTTP 接口**。

与标注训练平台 [train-center](https://github.com/1023l/train-center) 配合：那边产出 YOLO `.pt` 和 PaddleOCR inference，这边转成 TensorRT 后上线。

## 能做什么

- Web 演示：上传视频、拖拽计数线、正向/反向过线、L/R/成双/鞋码
- 算法 API：`POST /api/algo/process`（第三方软件每帧调一次）
- 推理后端：TensorRT 11 engine（主路径）或 ONNX Runtime

详细接口见 [docs/算法接口文档.md](./docs/算法接口文档.md)。

## 启动

```bash
# 建议 conda 环境 yolo-bench，依赖见 requirements-yolo-bench.txt
python web_server.py --host 0.0.0.0 --port 8001
```

- 页面：http://127.0.0.1:8001/
- Swagger：http://127.0.0.1:8001/docs
- 健康检查：http://127.0.0.1:8001/health

默认加载（需自行放到对应目录，权重不入库）：

```
models/fabric/fabric20260828V2.engine
models/text/text20260831V1.engine
models/rec/rec20260828V3.engine
```

rec 旁需保留 `inference.yml`（字符表），解码依赖它。

## 从 train-center 接到本仓库

1. 拿到 `fabric*.pt`、`text*.pt`、`rec*/`（Paddle inference 目录）
2. 在本机用 ultralytics / paddle2onnx 转 ONNX，再用 TRT 11+ 打 `.engine`
3. 拷进 `models/fabric|text|rec/`，改 `web_server.py` / `algo_routes.py` 顶部路径即可

运行时不需要 PyTorch / Paddle；转 engine 那一步需要。

## 依赖

- 轻量 ONNX：`requirements.txt`
- 现场 GPU + TRT：`requirements-yolo-bench.txt`（含 `tensorrt_cu12==11.2`、cupy）
