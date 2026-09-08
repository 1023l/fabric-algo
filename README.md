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

用 `tools/` 统一脚本（配置集中在 [tools/config.yaml](./tools/config.yaml)，换机器/出新版本只改它）：

```bash
# 1. 导出 ONNX（det 用 yolo-bench 环境，rec 自动切 paddle-ocr 环境）
python tools/export_onnx.py det --model all --version 20260828V2
python tools/export_onnx.py rec --version 20260828V3

# 2. 构建 TRT engine（trt_cache 计时缓存加速重建；日志/中间产物在 trt_export）
python tools/build_engine.py --model all

# 3. 验证 & 测速
python tools/verify.py engine          # 三 engine 加载+推理
python tools/verify.py onnx --model all
python tools/bench.py --model fabric

# 4. rec 精度（逐样本 预测 vs 标签）
python tools/eval_rec.py --backend onnx

# 5. 数据 / 接口辅助
python tools/crop_pieces.py --video xx.mp4 --out data_pieces/   # text 标注数据源
python tools/restore_det_images.py --dataset det_text           # det 数据自救
python tools/api_smoke.py                                       # 算法接口冒烟
```

运行时不需要 PyTorch / Paddle；`tools/` 的导出/构建那一步需要（环境自动切换）。

## 依赖

- 轻量 ONNX：`requirements.txt`
- 现场 GPU + TRT：`requirements-yolo-bench.txt`（含 `tensorrt_cu12==11.2`、cupy）
