# 鞋面检测 + OCR 推理性能优化记录

> 场景：现场载台速度 30m/min，相机对鞋面布片连续拍照，需实时完成
> **布片检测（yolo26）→ 文字区检测（yolo26）→ 文字识别（PaddleOCR rec）** 全流程。
> 优化目标：把单块布片 OCR 从 **5 秒** 降到几十毫秒，满足载台实时性。

---

## 一、初始基线（纯 CPU，onnxruntime）

| 环节 | 模型 | 耗时 |
|---|---|---|
| fabric 布片检测 | yolo26n（640） | 28 ms/帧 |
| text 文字区检测 | yolo26n（640） | 53 ms/片 |
| rec 文字识别 | PP-OCRv5_server_rec | **907 ms/张** |
| **单块布片 OCR 合计**（4~6 个文字区） | | **约 5 秒** |

**瓶颈**：rec 是 server 级大模型（PP-HGNetV2_B4 backbone + SVTR attention），
CPU 上 onnxruntime 对 attention 结构无专门加速，单张就要近 1 秒。

---

## 二、逐级优化（每一步都有实测数据）

### ① 减少模型调用：OCR 只在"过线"时触发
- 问题：原来每帧对每个布片都做完整 OCR
- 优化：只对**跨过计数线**的布片做 OCR（跨线当帧触发，12 帧窗口内重试，成功即停）
- 效果：OCR 调用次数从"每帧每片"降为"每片过线一次"

### ② 裁剪 rec 输入宽度：320 → 192
- 原因：rec 输入是 `(3, 48, 320)`，padding 到 320 浪费 2~3 倍计算
- 优化：`--rec-max-w 192`（鞋码类短文字足够），宽度自适应 `ceil(48*w/h)`
- 效果：rec 计算量直接减半以上

### ③ rec 并行：多 worker 独立 session 线程池
- 问题：共享同一 ORT session 并发调用有内部锁竞争（实测并行反而更慢）
- 优化：每个 worker 独立 session，`ThreadPoolExecutor` 按 stride 分片并行
- 额外：先试过 batch 推理（`recognize_batch`），因 ONNX **动态 shape 每次重建图反而更慢**，废弃

### ④ 上 GPU：onnxruntime CUDA EP
- 环境：RTX 4060 Laptop（现场目标 3090）
- 效果：

| 环节 | CPU | CUDA EP |
|---|---|---|
| fabric | 35 ms | **13 ms** |
| text | 53 ms | **25 ms** |
| rec | 907 ms | **19 ms**（47 倍） |

### ⑤ 终极加速：TensorRT engine（TRT 11）
- fabric/text（yolo26）和 rec 全部由 ONNX 构建 TRT engine
- 关键点：
  - **yolo26 需要 TRT 11+**（TRT 10.3 构建直接卡死——不认识 2025 新架构算子）
  - 不走 onnxruntime TRT EP（它绑定 TRT 10.x），改用 **tensorrt OnnxParser 直接从 ONNX 构建**，
    业务端运行时零依赖 torch / paddle
  - rec 是**动态宽**输入：OnnxParser + optimization profile（宽 16~320）
- 效果：

| 环节 | CUDA EP | **TRT engine** |
|---|---|---|
| fabric | 13 ms | **6.7 ms** |
| text | 25 ms | **5.6 ms** |
| rec | 19 ms | **2.1 ms**（动态宽 w=64） |

---

## 三、关键发现

1. **yolo26（2025 新架构）与 TRT 版本强绑定**：TRT 10.3 不支持，TRT 11+ 官方支持。
   网上说"TRT 用不了 yolo26"基本是版本太老。
2. **onnxruntime TRT EP ≠ 直接 TRT**：ORT 的 TRT EP 按 ORT 版本绑定 TRT 大版本（1.20.x 绑 10.x），
   想用新 TRT 就绕开它，直接用 tensorrt 的 OnnxParser 构建 engine。
3. **GPU DVFS 频率滞后**（容易被误判为"GPU 慢"）：
   - 逐帧视频处理时，帧间 GPU 空闲 → 掉到 P8 低功耗 → 下次推理要重新升频，单发多吃 10~15ms
   - 连续推理（30fps 视频流 / benchmark 循环）GPU 保持高频，无此问题
   - 测试视频逐帧处理实测：不锁频 25ms vs 锁频 2400MHz 后 14ms
   - 现场 RTSP 连续流 + 桌面卡（3090）基本不受影响；兜底方案：锁频 `nvidia-smi -lgc` 或软件保活线程
4. **业务端轻量集成**：运行时只依赖 `tensorrt + cupy + numpy + opencv`（rec 可回退 onnxruntime），
   完全不需要 torch / paddle / ultralytics。

---

## 四、最终结果（全 TRT，v9 实测）

| 环节 | 最终耗时 |
|---|---|
| fabric 布片检测 | 13.3 ms/帧 |
| text 文字区检测 | 11.5 ms/片 |
| rec 文字识别 | 4.0 ms/张（并行 4 worker） |
| **单块布片 OCR 合计** | **约 16 ms** |
| **整帧流水线（含过线帧）** | **约 29 ms** |

**对比初始 5 秒 → 29 ms，提速约 170 倍**，完全满足 30m/min 载台实时性。

---

## 五、部署建议（现场 3090）

```bash
# 模型文件（均已在业务侧验证）
models/fabric/fabric20260827V1.engine   # yolo26, 640, fp32
models/text/text20260827V1.engine       # yolo26, 640, fp32
models/rec/rec20260827V1.engine         # PP-OCRv5 server rec, 动态宽, fp32

# 推理命令（fabric/text/rec 全 TRT）
python infer_business.py --source <视频或RTSP> --ocr --provider trt \
    --fabric models/fabric/fabric20260827V1.engine \
    --text   models/text/text20260827V1.engine \
    --rec    models/rec/rec20260827V1.engine
```

- 现场连续视频流 GPU 保持高频，DVFS 影响小；3090 桌面卡散热好，性能优于本机 4060 数据
- 如需更极致：engine 可重新以 FP16 构建（本机为稳定性用 FP32，已达标）

---

## 六、技术栈

- **检测**：yolo26n（ultralytics 自训）→ ONNX → TRT 11 engine（OnnxParser）
- **识别**：PaddleOCR PP-OCRv5_server_rec → ONNX（paddle2onnx）→ TRT 11 动态宽 engine
- **运行时**：tensorrt + cupy + numpy + opencv（无 torch / paddle）
- **推理入口**：[infer_business.py](infer_business.py)（`--provider trt`）
- **引擎封装**：[onnx_engine.py](onnx_engine.py)（TrtOnnxDetector / RecTrtEngine，接口与 ONNX 版一致，可无缝切换）
