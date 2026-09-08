# 多 GPU / 分布式训练 / 集群方案 · 技术选型备忘

> 适用对象：`train-center`（布料 OCR 标注训练平台）未来需要扩展到**单机多卡**和**多机集群**时的落地路线图。  
> 本文档**不入库 train-center**，仅作为本地技术总结。日期：2026-09-01。

---

## 0. 现状

我们当前是**单机单卡（1× RTX 4060）**：
- det_fabric / det_text：`ultralytics YOLO26`，默认 `device=0`，单卡跑，69 张图 100 epoch 几分钟；
- rec_text：`PaddleOCR/tools/train.py`，单卡，`Global.use_gpu=true`，约 200 epoch 十几分钟。

**为什么当前没有立刻要做多 GPU？** 数据量是瓶颈（rec 训练 61 张/78 标签），模型即使给 8 卡通信开销反而比收益大。**但技术路线要预先搭好**：一旦真实业务把数据扩到万张级，训练时长将成为关键路径。

---

## 1. 单机多卡（一台机器多张 GPU）

### 1.1 PyTorch 侧：det 两阶段（YOLO26 ultralytics）

ultralytics 天然支持 **DDP（DataDistributedParallel）**。

| 方式 | 命令 | 场景 |
|---|---|---|
| 单卡 | `yolo train model=... device=0` | 调试 / 小数据 |
| 多卡 DDP | `yolo train model=... device=0,1,2,3` | 单机 4 卡（总 batch 会被 world_size 均分，每张卡的 micro_batch 由脚本自动算） |
| 自定义 launcher（显式 torchrun） | `torchrun --nproc_per_node=4 train.py --fabric ...` | 我们自定义的 `train.py` 里加 DDP bootstrap 分支 |

**DDP 原理**：每个 rank 持有独立模型副本 + **独立数据分片**（`DistributedSampler`），前向/反向完通过 `all-reduce` 聚合梯度，更新全同步。

**坑位（我们代码要改的地方）**
1. `core/export.py` 生成的 `data.yaml` 不变；`train.py` 启动时要 `init_process_group` 并把 `DataLoader sampler` 换成 `DistributedSampler`（ultralytics 已经封装好了，只要给 `device=0,1,2,3` 它自动做）。
2. **batch_size 语义变化**：我们传的 `batch=8` 在 DDP 下指**每张卡 8**，总 global_batch = 8×N；若我们把 batch 解释为总 batch，要做 `per_gpu_batch = batch / N` 并记录到 args.yaml。
3. 保存 checkpoint 只在 `rank==0` 保存（否则 8 个 rank 同时写 best.pt 会冲突；ultralytics 默认已这么做）。
4. NCCL 在 Windows 支持不如 Linux，若以后上服务器推荐 **Ubuntu 22.04**；Windows 用 gloo 后端替代也行但慢 20%。

**加速建议（比 baseline 单卡）**：
- 2 卡 ≈ 1.7x；4 卡 ≈ 3.1x；8 卡 ≈ 5.5x（YOLO26n 单卡迭代已经快，通信占比上升）。

### 1.2 PaddlePaddle 侧：rec 阶段（PP-OCRv5）

Paddle 原生叫 **Fleet**，API 已经内嵌到 `PaddleOCR/tools/train.py`。只要把 `Global.distribute=true` 并在命令行加 GPU 数，Paddle 自动拉起 `parallel_run`。

```bash
# 单机 2 卡（推荐）
python -m paddle.distributed.launch --gpus="0,1" tools/train.py -c configs/rec/rec_svtrnet_ch.yml -o Global.use_gpu=true Global.distribute=true
```

- 通信默认 **NCCL 后端**（GPU 之间 P2P）；
- **global_batch = per_device_train_batch_size × N_GPUS**，和 PyTorch 一样把 batch 摊到每张卡；
- 保存同样只在 `trainer.rank==0` 写 `best_accuracy.pdparams`。

**rec 的收益估算（我们 server config，batch 8，imgsz 3×48×320）**：
- 2 卡 ≈ 1.6x，4 卡 ≈ 2.7x。

---

## 2. 多机多卡（集群）

### 2.1 PyTorch：torchrun + DDP（2~8 机最常见）

每台机器起同样一条命令，只需指定 `nnodes`、`node_rank`、`master_addr:master_port`：

```bash
# 机器 0（master）
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
         --master_addr=10.0.0.1 --master_port=29500 train.py ...

# 机器 1（worker）
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
         --master_addr=10.0.0.1 --master_port=29500 train.py ...
```

**带宽要求**：DDP 对跨机通信敏感，要求至少 **10 Gbps 以太网**；若用 **InfiniBand/RDMA**（200Gbps）才建议跨机 ≥8 机，否则 all-reduce 会把收益吃光。

### 2.2 Paddle：Fleet Multi-Node

```bash
# 写 node_ip_list.txt
echo "10.0.0.1 slots=8" > ips
echo "10.0.0.2 slots=8" >> ips

# master 启动
python -m paddle.distributed.launch --servers="10.0.0.1,10.0.0.2" \
    --gpus=0,1,2,3,4,5,6,7 --ips=ips tools/train.py -c configs/rec/rec_svtrnet_ch.yml
```

### 2.3 跨机的两个大杀器

| 痛点 | 技术 | 作用 |
|---|---|---|
| 模型太大装不下一张卡（10B+） | **ZeRO (DeepSpeed / FSDP)** | ZeRO-1/2/3：把 optimizer state / gradient / param 分片到各 rank，降低单卡显存占用 **3-8 倍** |
| 数据在共享盘 IO 瓶颈 | **分布式文件系统（Lustre/NFSv4.2/Alluxio）** | 小文件（crop_img 里单张 30-50KB 万级）先打包成 tar/webdataset，再按 node 本地 cache |

**DeepSpeed / FSDP 选型**：
- DeepSpeed（微软）：ZeRO-3 + offload 参数可把 **70B 模型训在 8×4090**（用 CPU NVMe 卸载）。对我们 YOLO26 / PP-OCRv5（<100M 参数）完全杀鸡用牛刀，暂不优先。
- PyTorch **FSDP**（Fully Sharded Data Parallel）：原生内置，2023 后更稳。如后续做超大 VLM（布片多模态检测，ViT-G/1B 参数）直接上。

---

## 3. 超算 / 集群调度

多机多卡手动 torchrun 太烦，生产都用**任务调度器**批量排队，常见 3 种：

### 3.1 Slurm（学术/国产超算标配）
- 写一个 `sbatch train.sh`，指定 `--gres=gpu:8 --nodes=2 --ntasks-per-node=8`
- 调度器分配节点、注入 `SLURM_JOB_NODELIST / SLURM_PROCID` 环境变量；torchrun/Paddle 能直接读
- **和 train-center 的对接点**：我们 `server/routes/train_routes.py` 新增一条 `submit_slurm` 分支，把本地 Popen 换成 `sbatch --parsable train.sh`，记录 job_id，然后 10s 轮询 `squeue -j <id>` + `sacct` 看日志

### 3.2 K8s + Kubeflow Training Operator / Volcano
- 现在云厂**自研超算/AI 平台**基本都在 K8s 上：PyTorchJob / TFJob / PaddleJob（Training Operator 统一管理）
- 用 `PVC` 挂载数据集 / 模型输出，用 `Istio` 暴 `tensorboard` 端口
- 对我们平台的**改造点**：把 train_routes 里的 Popen 改成调 K8s Python client `create_namespaced_custom_object`，状态轮询换成 `list_namespaced_custom_object_status`

### 3.3 国产超算：Sunway / 天河 / 曙光
- 通常是 Slurm 的定制版 + 作业预占 + GPU/MIG 分片
- 额外坑：**编译器/MPI 是自研**（如 Sunway swgcc），pytorch/paddle 要在平台交叉编译再提交。我们业务代码基本不碰，只要在 train.py `trainer` 里加一个 `--backend=gloo` fallback 即可。

---

## 4. 性能调优清单（未来我们实际落地时按顺序拉通）

| 层级 | 项 | 预期提升 | 备注 |
|---|---|---|---|
| **数据 IO** | 小文件打包（WebDataset / LMDB） | +20-50% | rec crop_img 1万张时最明显 |
| | NVMe 本地 cache（每训练机 1TB 作 cache 盘） | +10-30% | 避免共享盘读放大 |
| | DataLoader `num_workers=2*N_GPU + pin_memory=True` | +10-15% | 当前默认 num_workers 可能偏低 |
| **混合精度** | AMP（FP16/BF16）| +30-70% | ultralytics 默认开；PaddleOCR 要在 `Global.fp16_loss_scaling` 打开 |
| | TF32（Ampere+：A10/30/100） | +15-30% | torch.backends.cuda.matmul.allow_tf32 = True |
| **通信** | 梯度累积（global_batch 不够时） | 间接提升准确率 | global_batch = per_gpu × N × accum |
| | 通信 bucket 调优（DDP bucket_cap_mb 25→100） | +5-10% | 大模型收益大 |
| | NVLink / InfiniBand + GPUDirectRDMA | 跨机 +100-300% | 硬件层 |
| **内存** | gradient checkpointing | -30% 显存，-15% 速度 | 显存不够用 |
| | ZeRO-2 (FSDP stage 2) / DeepSpeed Stage 2 | -50% 显存 | 我们不需要，直到模型>1B |
| | FlashAttention 2/3 | +50% 速度（纯 attention 部分） | rec SVTR/ViT 分支可接 |
| **调度** | 任务冷启动：镜像懒拉取 + Nydus | 排队时间 → 0 | K8s 场景 |
| | MIG 切小卡（如 A100 1g.5gb → 跑 7 个小实验） | 利用率 2-3x | 多实验并行 |

---

## 5. 监控 / 可视化

不管单机还是集群，都要统一：

1. **训练指标**：`TensorBoard` 或 `wandb/MLflow`。ultralytics 已经写了 `results.csv / BoxPR_curve`，PaddleOCR 有 `train.log` — 我们 `train-center` 的 "日志" 页已经在展示 log；下一步可以加一个 "曲线" Tab，直接读 `results.csv` 或 `event.out.*` 用 ECharts 画。
2. **GPU 利用率**：每 10s 采一次 `nvidia-smi dmon -s u`，若 GPU-Util < 60% 说明 IO/CPU 是瓶颈，加大 `num_workers` / 打包小文件 / 用 DALI 解码。
3. **告警**：任务 fail / OOM / eval acc 3 epoch 不涨 → 飞书/企业微信 webhook，不用蹲前端。
4. **成本**：A100 80GB × 8 节点≈¥50/h，训练前跑一个 100-iter 基准估 ETA，防止写错超参把几万块跑飞。

---

## 6. 落地路线（按优先级）

| 阶段 | 改什么 | 预计工作量 | 收益 |
|---|---|---|---|
| **P0（本周内）** | `train.py` 支持 `--device 0,1,..N`（ultralytics 自带，其实不用改啥，只是 train-center API 入参暴露 device 字段）；`train_rec.py` 支持 `--gpus 0,1,..N` 透传到 `paddle.distributed.launch` | 0.5 人日 | 单台双卡机器立即提速 1.6-1.7x |
| **P1（下个月数据到 1000+）** | Slurm / K8s 二选一（按现场环境），train_routes 里抽象一个 `TaskBackend`（local_popen / slurm_sbatch / k8s_pytorchjob），状态轮询统一 | 2-3 人日 | 可跑在 4×8 卡集群上，总训练时长降 1/4 |
| **P2（数据 10000+ / 大模型分支）** | DeepSpeed ZeRO-2 + FSDP；小文件转 WebDataset；TensorBoard/MLflow 面板；wandb 扫参（贝叶斯超参搜索 lr / weight_decay / iou / conf） | 5-7 人日 | 稳定训大模型；一个 sweep 任务顶手工调 50 次 |
| **P3（可选，上国产超算）** | Sunway/天河适配；C/CUDA kernel 用 swclang 转；MPI 通信替换 | 1+ 人月 | 合规（国产化）+ 大规模算力 |

---

## 7. 与 train-center 具体代码的对接点

未来修改集中在 3 个文件：

- **`server/routes/train_routes.py`**
  - 新增训练启动参数：`devices="0,1,2,3"` / `nnodes` / `backend`
  - 抽象 `TaskBackend`：`LocalPopenBackend`（现在的实现）/ `SlurmBackend` / `K8sBackend`
  - `/api/train/tasks/:id` 的 snapshot 额外返回 `slurm_job_id / k8s_pod_name / gpu_util / iter_smoothed`

- **`train.py`（det）**
  - ultralytics `model.train(device=devices)` 一行就能多卡；关键是 **在 DDP rank 0 时把 metrics.json 写到我们 models/det/ 目录**，其他 rank return。
  - `DistributedSampler.set_epoch(epoch)` 保证每个 epoch 数据 shuffle 方式不一样（ultralytics 封装了，但我们自定义的 det augmentation 要同步）。

- **`train_rec.py`（rec）**
  - 当 `len(gpus)>1` 时把 `python tools/train.py` 换成 `python -m paddle.distributed.launch --gpus=... tools/train.py`，其余 `-o Global.use_gpu=true Global.distribute=true` 参数不变。
  - 训练完同样只在 master rank 调 `tools/export_model.py`，并把 inference 复制到 models/ocr/rec_xxx。

- **`core/cleanup_artifacts.py`**
  - DDP 产生的多机 checkpoint 会在 `output/<jobname>/node<rank>/iter_epoch_*`，扫描函数加一条递归 `rglob()` 就能覆盖（当前已支持），清理流程不用改。
