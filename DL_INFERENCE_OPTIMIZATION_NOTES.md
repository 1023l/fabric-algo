# 深度学习模型推理性能优化学习笔记

> 结合本项目（yolo26 检测 + PaddleOCR 识别，5s → 29ms）的实战案例，
> 系统梳理 GPU 推理加速的知识体系。适合已有基础 CUDA 概念的读者。

---

## 1. 推理与训练的本质差异

| | 训练 | 推理 |
|---|---|---|
| 计算 | 前向 + 反向 + 梯度更新 | 只有前向 |
| Batch | 大 batch（32/64/128） | 通常 batch=1（流式/逐帧） |
| 指标 | 吞吐（样本/s）、收敛 | **延迟（ms/帧）**、吞吐 |
| 运行库 | PyTorch / Paddle 完整框架 | 轻量引擎（ORT / TRT） |
| 硬件 | GPU 满载 | 可能间歇性空闲 |

**推理优化的核心目标**：最小化单次推理延迟 + 最大化吞吐，同时保持精度。

**关键认知**：推理瓶颈往往不在"算力"而在**内存带宽、kernel 启动开销、数据搬运、库的额外开销**。这就是为什么"模型小却慢"很常见。

---

## 2. CUDA 基础（温故）

### 2.1 GPU 硬件结构

```
GPU（显卡）
└── 多个 SM（Streaming Multiprocessor，流式多处理器）
    ├── CUDA core（负责算术运算，如 FP32/INT8）
    ├── Tensor Core（矩阵乘/卷积专用，FP16/INT8 加速，AI 核心）
    ├── 共享内存（Shared Memory，SM 内，快）
    └── 寄存器堆（Register File）
├── L2 Cache（跨 SM 共享）
└── 显存（Global Memory，最大最慢）
```

- **GPU 适合神经网络的原因**：矩阵乘、卷积都是"同一条指令作用于大量数据"——
  大规模并行 + 高显存带宽恰好对路。
- **Tensor Core**：从 Volta 架构引入，专门加速 FP16/INT8 矩阵运算，是 AI 推理的核心硬件。

### 2.2 执行模型

- **Kernel**：一次 GPU 函数调用
- **Grid → Block → Thread**：线程组织层级
- **Warp = 32 线程**：GPU 以 warp 为单位调度（SIMT，单指令多线程）
  - **Warp divergence**：warp 内分支分歧会导致串行执行（`if/else` 对同一 warp 内不同线程）
- **Occupancy**：SM 上活跃线程数占比，越高越能**隐藏延迟**
- **延迟隐藏**：GPU 靠大量并行线程掩盖访存延迟（等数据时切换执行别的 warp）

### 2.3 内存访问

- **显存带宽是硬约束**：一次 H2D/D2H 拷贝比 kernel 本身可能还贵
- **Coalescing（合并访问）**：同一 warp 尽量访问连续地址，一次事务取回
- 对推理的启示：**减少 H2D/D2H 拷贝、数据连续布局**往往比减少计算量更有效

---

## 3. 推理优化层次总览

优化的手段分布在多个层面，**从上到下成本递减、收益递增**：

```
┌─────────────────────────────────────────────┐
│ 系统/工程层：预热、并行、流水线、缓存、DVFS    │  最容易，收益大
├─────────────────────────────────────────────┤
│ 推理引擎层：TensorRT、onnxruntime EP、OpenVINO │  图优化 + 算子融合 + 自动调优
├─────────────────────────────────────────────┤
│ 算子层：kernel 融合、cuDNN/cuBLAS 选择        │
├─────────────────────────────────────────────┤
│ 模型层：轻量结构、剪枝、蒸馏、量化（FP16/INT8） │  改动模型，需重训/微调
└─────────────────────────────────────────────┘
```

---

## 4. 模型层面优化

### 4.1 轻量网络结构
- MobileNet / ShuffleNet / PPLCNetV3（Paddle）：深度可分离卷积，参数量小一个量级
- yolo26n（nano）：ultralytics 轻量检测模型
- **本案例**：server 版 rec（PPHGNetV2_B4）→ 若换 mobile 版（PPLCNetV3），CPU 上快 5~10 倍

### 4.2 剪枝（Pruning）
- 去掉不重要的通道/权重（结构化剪枝），模型变小变快，精度损失小，需微调

### 4.3 知识蒸馏（Distillation）
- 大模型（teacher）教小模型（student），小模型逼近大模型精度

### 4.4 量化（Quantization）
- **FP16**：精度减半（2 字节），Tensor Core 原生支持，精度几乎无损，显存减半
- **INT8（PTQ）**：8 位整型，需**校准数据集**统计激活范围，推理快 2~4 倍，精度可能有损
- **QAT**：训练时模拟量化，精度更好
- **本案例**：当前 engine 是 FP32（TRT 11 的 FP16 flag 写法不同，未开启）；实际 FP16 一般可再快 ~2 倍

---

## 5. 推理框架：onnxruntime 与 EP

### 5.1 什么是 EP（ExecutionProvider）

onnxruntime 是"前端统一 ONNX 图，后端可插拔"的架构：

```
ONNX 模型
  └── ORT 优化（图重写、常量折叠）
        └── 选择一个 EP 执行算子
              ├── CPUExecutionProvider（通用）
              ├── CUDAExecutionProvider（NVIDIA GPU，cuDNN/cuBLAS）
              ├── TensorrtExecutionProvider（NVIDIA，把图交给 TRT）
              ├── OpenVINOExecutionProvider（Intel CPU/GPU/VPU）
              └── ...
```

- **CUDA EP**：每个算子调用 cuDNN/cuBLAS，是"通用 GPU 加速"，开发简单、兼容好
- **TRT EP**：整图交给 TensorRT 做端到端优化（见第 6 节）
- **EP 是逐算子 fallback**：某算子 EP 不支持时回退下一级 EP（如 CPU），所以要显式给 providers 顺序

### 5.2 本案例的 EP 数据（4060）

| 环节 | CPU | CUDA EP | TRT |
|---|---|---|---|
| yolo26n 检测 | 35ms | 13ms | 6.7ms |
| rec（server 版） | 907ms | 19ms | 2.1ms |

> 有意思：轻量 yolo 在 CPU 上 35ms 但 CUDA 上 13ms——GPU 上算子已有 cuDNN 优化；
> rec 是 attention/LSTM 结构，CPU 没有专门加速，GPU 提升 47 倍。

### 5.3 为什么"ORT TRT EP"绑死 TRT 版本（本案例踩坑）

- ORT 编译时**链接特定 TRT 大版本**的库（如 1.20.x 找 `nvinfer_10.dll`）
- 装 TRT 11 后 ORT 报"找不到 nvinfer_10.dll" → 只能降级 TRT 10.3
- **结论**：想用新 TRT，绕开 ORT TRT EP，直接用 TensorRT 的 OnnxParser 构建 engine

---

## 6. TensorRT 深入

### 6.1 是什么

NVIDIA 官方推理引擎（C++/Python），针对**部署场景**做极致优化：

1. **图优化**：算子融合（conv+BN+ReLU → 单 kernel）、常量折叠、死代码消除
2. **kernel 自动选择**：对每个算子从 kernel 库里按目标 GPU 自动调优（autotuning）
3. **精度支持**：FP32 / FP16 / INT8（PTQ 需校准）
4. **动态 shape**：同一 engine 支持多种输入尺寸（optimization profile）

### 6.2 engine 构建流程

```
ONNX ──OnnxParser──▶ Network（计算图）
                        │
            Builder（builder config：FP16/INT8、workspace、profile）
                        │
                   build_serialized_network()
                        │
                     engine（序列化 .engine 文件，可跨机部署，但绑 GPU 型号）
                        │
              Runtime.deserialize_cuda_engine() + execute
```

**本案例**（TRT 11，Python API）：
```python
builder = trt.Builder(logger)
network = builder.create_network()            # TRT 10+ 默认 EXPLICIT_BATCH
parser = trt.OnnxParser(network, logger)
parser.parse(onnx_bytes)                       # 解析 ONNX
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 构建内存上限
profile = builder.create_optimization_profile()   # 动态 shape 用
profile.set_shape(in_name, min=..., opt=..., max=...)
config.add_optimization_profile(profile)
engine = builder.build_serialized_network(network, config)  # 构建（可能几分钟）
```

### 6.3 动态 shape（Dynamic Shapes）

- 模型输入维度可以是 `-1`（动态）
- 构建时用 **optimization profile** 给出 `min / opt / max`，三个维度
  - `opt` 附近的性能最好，推理时每次设 `set_input_shape`
- **本案例**：rec 输入 `(1, 3, 48, 动态宽)`，profile `min=(1,3,48,16) opt=(1,3,48,64) max=(1,3,48,320)`

### 6.4 常见坑（本案例实录）

| 坑 | 现象 | 解决 |
|---|---|---|
| **新模型算子不支持** | yolo26（2025 Mamba 混合架构）在 TRT 10.3 构建卡死 | 升级 TRT 11+（官方已适配） |
| **动态 shape 批量推理慢** | 每次 shape 变化 ORT 重建执行图，batch 反而更慢 | 放弃 batch，单张并行 |
| **engine 绑 GPU 型号** | A 卡构建的 engine 不能在 B 卡跑 | 现场机器上重新构建（或按型号构建） |
| **首次构建慢** | 自动调优 + 层融合可能几分钟 | 构建一次后序列化缓存；部署时用 trtexec/API 一次性生成 |

---

## 7. GPU 推理工程细节（容易被忽略的"隐形开销"）

### 7.1 冷启动与预热（Warmup）
- 首次推理包含 kernel 编译、显存分配、cuDNN/TRT 初始化，可能慢几十倍
- **预热**：正式处理前跑 2~3 次 dummy 输入

### 7.2 H2D / D2H 拷贝
- CPU→GPU 传输入、GPU→CPU 取输出是**显式拷贝**，带宽有限
- 输入图像大（如 4024×3036）→ letterbox 到 640 后在 CPU 做，再传 640×640（4.9MB）
- 优化：**尽量在 GPU 端做预处理**（如 cuDNN 的图像处理、TensorRT 的 preprocess 插件），或异步拷贝（cudaMemcpyAsync + 双缓冲）

### 7.3 批量 vs 并行
- **固定 shape 时**：batch 推理通常比单张快（kernel 摊销）
- **动态 shape 时**：batch 会导致图重建/调度开销，**反而更慢**（本案例实测，已废弃 batch 方案）
- 并行方案：**线程池 + 每线程独立 session/context**（本案例 4 worker 并行 rec）

### 7.4 GPU 频率（DVFS）——本案例最大"隐形杀手"
- GPU 空闲时自动降到最低功耗（P8，~200MHz / 2W）
- 收到任务**升频需要时间**（几十 ms），任务结束又掉回
- 逐帧视频处理：每帧之间 GPU 空闲（CPU 在读图/绘制/写视频）→ 每次推理吃满升频滞后
- **实测**：不锁频 fabric 25.5ms vs 锁频 2400MHz 后 14.1ms
- **解决**：
  - 连续流（30fps RTSP）GPU 保持高频，影响小
  - 锁频：`nvidia-smi -lgc 2400`（恢复 `-rgc`）
  - 软件保活：后台线程周期性跑 dummy 推理保持 GPU 活跃
  - 桌面卡（3090）散热好、滞后远小于笔记本卡

### 7.5 测量方法
- **分层测时**：把"检测/检测/识别"各段分别计时（本案例三段 timing），定位瓶颈
- 去掉一次性开销（加载、预热）后再统计稳定值

---

## 8. 性能分析与调优工具

| 工具 | 用途 |
|---|---|
| `nvidia-smi` | 温度/频率/功耗/显存/利用率，判断是否降频、占用 |
| `nvidia-smi -lgc` | 锁定 GPU 频率（测试/部署用） |
| Nsight Systems | 全局时间线：kernel、拷贝、CPU-GPU 同步 |
| Nsight Compute | 单 kernel 微观分析：occupancy、带宽、利用率 |
| `trtexec` | TRT 命令行：构建 engine、benchmark、精度对比 |
| ORT Profiling | `sess_options.enable_profiling` 看每个算子耗时 |
| CPU 热点 | py-spy / cProfile（前处理、后处理、解码往往是隐藏热点） |

---

## 9. 本案例实战复盘（5s → 29ms）

| 步骤 | 手段 | 所属层次 | 单块布片 OCR |
|---|---|---|---|
| 基线 | CPU + ORT | - | ~5s |
| ① | OCR 只在过线触发 | 系统/工程 | 调用次数 ↓90% |
| ② | rec 输入宽 320→192 | 算子/系统 | 计算量减半 |
| ③ | 多 worker 并行 rec | 系统/工程 | 并行提速 |
| ④ | GPU（CUDA EP） | 推理引擎 | rec 907→19ms |
| ⑤ | TRT 11 engine | 推理引擎 | rec→2.1ms |
| 最终 | 全 TRT | - | ~16ms（整帧 ~29ms） |

**结论**：从"减少无用计算"（①）→"减小输入"（②）→"并行"（③）→"换硬件执行"（④）→"专用推理引擎"（⑤），**层层递进，每层都有独立收益**。优化的正确顺序就是先找瓶颈、再对症下药，而不是盲目堆技术。

---

## 10. 延伸学习建议

1. **CUDA 编程**：《CUDA C++ Programming Guide》（NVIDIA 官方）、Udacity CUDA 课程
2. **TensorRT 官方**：TensorRT Developer Guide + 示例（samples/ 里的 onnx_resnet、dynamic shapes 例子）
3. **Nsight**：NVIDIA 官方 profiling 教程（看懂时间线是调优基本功）
4. **INT8 量化**：TensorRT PTQ 校准、nvidia-modelopt（本案例中 ultralytics 导出 engine 用到的工具）
5. **推理服务化**：Triton Inference Server（并发调度、模型管理）
6. **大模型推理**：vLLM / TensorRT-LLM（KV cache、paged attention、continuous batching——把本节思想放大到 LLM）
7. **混合精度**：NVIDIA AMP（automatic mixed precision）文档

> 一句话总结：**推理优化 = 减少计算 + 减少搬运 + 选对执行引擎 + 管好硬件状态**，
> 先用 profiling 找到瓶颈，再逐层优化，每一步都能量化。
