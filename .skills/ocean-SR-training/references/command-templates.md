# 训练命令模板（超分辨率）

> 版本: 1.0.0 | 最后更新: 2026-03-04
>
> **重要**：正常情况下训练必须通过 `ocean_sr_train_start` 工具启动。
> 本文档仅供工具不满足需求时参考正确的命令模板。

---

## Python 环境

### 正确路径

训练工具内部使用 `findPythonWithModule('torch')` 自动查找含 PyTorch 的 Python。

手动验证当前环境：
```bash
# 查看工具会使用的 Python 路径
/home/lz/miniconda3/envs/pytorch/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 禁止路径

以下路径 **不含 PyTorch**，禁止用于训练：

| 路径 | 问题 |
|------|------|
| `/home/lz/miniconda3/bin/python` | base 环境，无 torch |
| `/usr/bin/python3` | 系统 Python，无 torch |
| `/usr/local/bin/python3` | 系统 Python，无 torch |
| `python` / `python3`（未指定完整路径） | 可能解析到错误环境 |

---

## 训练启动模板

> **所有模板仅供参考**。正常流程应使用 `ocean_sr_train_start` 工具。

### 单卡训练

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/lz/miniconda3/envs/pytorch/bin/python \
  {log_dir}/_ocean_sr_code/main.py \
  --mode train \
  --config {log_dir}/config.yaml
```

### DP 多卡训练（Data Parallel）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/home/lz/miniconda3/envs/pytorch/bin/python \
  {log_dir}/_ocean_sr_code/main.py \
  --mode train \
  --config {log_dir}/config.yaml
```

### DDP 多卡训练（Distributed Data Parallel）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29500 \
/home/lz/miniconda3/envs/pytorch/bin/python \
  -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port=29500 \
  {log_dir}/_ocean_sr_code/main_ddp.py \
  --mode train \
  --config {log_dir}/config.yaml
```

**DDP 关键点**：
- 使用 `torch.distributed.run`（不是 `torch.distributed.launch`）
- `--nproc_per_node` = GPU 数量
- `--master_port` 必须是空闲端口
- 环境变量 `CUDA_VISIBLE_DEVICES` 必须设置
- launcher 自动设置 `RANK`, `LOCAL_RANK`, `WORLD_SIZE`

---

## 禁止命令

| 命令 | 原因 | 正确替代 |
|------|------|---------|
| `python -m torch.distributed.launch ...` | **已废弃**，不设置 `LOCAL_RANK` 环境变量 | `python -m torch.distributed.run ...` |
| `python main_ddp.py --mode train ...` | 缺少 `RANK`/`LOCAL_RANK`/`WORLD_SIZE` 环境变量 | 必须通过 `torch.distributed.run` 启动 |
| `python main.py ...`（无 `CUDA_VISIBLE_DEVICES`） | GPU 分配不确定 | 始终显式设置 `CUDA_VISIBLE_DEVICES` |
| 使用 `python` / `python3` 无完整路径 | 可能解析到无 torch 的环境 | 使用完整路径或通过训练工具启动 |

---

## 调试命令

### GPU 状态

```bash
# 查看 GPU 使用情况
nvidia-smi

# 持续监控 GPU（每 2 秒刷新）
nvidia-smi -l 2

# 仅查看 GPU 显存使用
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

### NCCL 调试

```bash
# 开启 NCCL 详细日志
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 \
/home/lz/miniconda3/envs/pytorch/bin/python \
  -m torch.distributed.run --nproc_per_node=2 ...
```

### 日志查看

```bash
# 查看训练日志最后 100 行
tail -n 100 {log_dir}/train.log

# 查看错误日志
cat {log_dir}/train-*.error.log

# 实时跟踪日志
tail -f {log_dir}/train.log
```

### 端口检查

```bash
# 检查 DDP 端口是否被占用
lsof -i :29500

# 查找可用端口
for port in $(seq 29500 29599); do
  lsof -i :$port > /dev/null 2>&1 || { echo "可用端口: $port"; break; }
done
```

### CUDA 错误详细堆栈

```bash
# 设置同步执行以获取精确错误位置
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 \
/home/lz/miniconda3/envs/pytorch/bin/python \
  {log_dir}/_ocean_sr_code/main.py \
  --mode train \
  --config {log_dir}/config.yaml
```
