# fabric-algo

通用视觉检测推理端：**目标检测 + OCR 识别 + 实时计数 + 第三方 HTTP 接口**。

> 项目定位是一套与具体行业解耦的检测识别流水线（检测目标 → 区域文字识别 → 跟踪计数），
> 当前以**纺织布片计数 / 鞋码识别**作为参考实现。仓库沿用 `fabric-algo` 名称，
> 模型/数据集命名（`fabric` / `text` / `rec`）为数据集代号，不绑定行业。

与标注训练平台 [train-center](https://github.com/1023l/train-center) 配合：那边产出 YOLO `.pt` 和 PaddleOCR inference，这边转成 TensorRT 后上线。

## 能做什么

- Web 演示：上传视频、拖拽计数线、正向/反向过线计数、目标属性（L/R / 成双 / 识别文本）
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

## 模型管理（上传即上线）

页面底部「模型管理」支持两种上传方式：

```
POST /api/models/upload         # 整包：model_bundle_*.zip（训练平台「一键导出3个模型」产物）
POST /api/models/upload_single  # 单模型：fabric/text 传 .pt；rec 传 rec*.zip 或模型文件夹
GET  /api/models/task           # 转换进度（导出 ONNX -> TRT engine -> 热替换）
GET  /api/models/current        # 当前生效版本（models/current.json）
```

- **整包**：3 个模型一起替换；**单模型**：只转换上传的那个，其余保持当前版本
- 单模型命名约定：`fabric20260910V1.pt` / `text20260910V1.pt` / rec 文件夹 `rec20260910V1/`（须含 best.pdparams）
- 后台自动完成 ONNX 导出、TRT engine 构建，并**先试加载、成功后热替换**在线模型（失败则旧模型继续服务）
- 上传版本须不同于当前生效版本（防重复，如需重建请改版本号）

## 依赖

- 轻量 ONNX：`requirements.txt`
- 现场 GPU + TRT：`requirements-yolo-bench.txt`（含 `tensorrt_cu12==11.2`、cupy）
