# 错误处理指南

> 版本: 3.3.0 | 最后更新: 2026-02-09

遇到错误时，必须展示：**错误信息 + 可能原因 + 修改建议**

---

## 数据相关错误

### FileNotFoundError: HR/LR directory not found

**可能原因**：
- 数据目录路径不正确
- 预处理尚未执行或未完成
- 变量名与预处理输出不匹配

**修改建议**：
1. 检查数据目录：`ls <dataset_root>/train/hr/`
2. 确认变量名匹配预处理输出
3. 重新执行预处理或提供正确路径

### HR/LR file count mismatch

**可能原因**：
- 预处理过程中断，HR 和 LR 文件数不一致
- 手动删除了部分文件

**修改建议**：
1. 检查两个目录的文件数量
2. 重新执行预处理生成完整数据

### No .npy files found

**可能原因**：
- 目录存在但为空
- 文件扩展名不是 `.npy`

**修改建议**：
1. 确认目录中有 `.npy` 文件
2. 检查预处理是否正常完成

---

## GPU 相关错误

### CUDA out of memory

**可能原因**：
- Batch size 过大
- 模型参数量超出显存
- 其他进程占用 GPU 显存
- 高分辨率全图训练显存不足

**修改建议**：
1. 启用 AMP 混合精度 `use_amp=true`（减少约 40-50% 显存，最易操作）
2. 减小 batch_size（如 8 → 4 → 2）
3. 启用梯度检查点 `gradient_checkpointing=true`（减少约 60% 激活显存；当前默认已开启，可显式确认）
4. 设置 `patch_size`（如 64 或 128）裁剪小区域训练
5. 使用多卡训练分摊显存
6. 用 `ocean_sr_check_gpu` 查看显存占用情况
7. 选择更轻量的模型

**显存预估失败（OOM）**：
- 训练前的显存预估阶段已检测到 OOM
- 系统会自动尝试降级：开启 AMP（若当前关闭）→ 减半 batch_size（最多 5 次）
- 所有自动优化手段耗尽后才报错，并建议使用更大显存 GPU 或设置 patch_size

### NCCL error (DDP 模式)

**可能原因**：
- GPU 之间通信失败
- 指定的 GPU 不可用

**修改建议**：
1. 检查指定的 GPU 是否都可用
2. 尝试使用 DP 模式替代 DDP
3. 减少使用的 GPU 数量

---

## 训练相关错误

### Loss 为 NaN

**可能原因**：
- 学习率过大导致梯度爆炸
- 数据中包含异常值
- 归一化配置不当

**修改建议**：
1. 降低学习率（如 0.001 → 0.0001）
2. 检查数据是否包含 NaN/Inf
3. 尝试不同的归一化方式（PGN → GN）

### 训练不收敛（Loss 不下降）

**可能原因**：
- 学习率过小
- 模型容量不足
- 数据量不够

**修改建议**：
1. 增大学习率
2. 选择更大的模型
3. 增加训练数据

---

## 模型相关错误

### Model not implemented / 模型未接入

**可能原因**：
- 模型名称拼写错误
- 模型未在注册表中
- 模型目录存在但未接入训练流程（缺少注册、Trainer 或模板）

**修改建议**：
1. 调用 `ocean_sr_list_models` 查看正确的模型名
2. 仅选择 `supported=true` 的模型
3. 若要接入新模型，补齐 model 注册、trainer 映射与模板配置

### FFT + AMP 可能不兼容

**可能原因**：
- FNO2d/HiNOTE/MWT2d/M2NO2d/MG-DDPM 等 FFT/频域模型在特定尺寸下开启 AMP 会触发 cuFFT 限制
- LR 高宽不是 2 的幂或 patch_size 与 scale/divisor 不匹配

**修改建议**：
1. 保持默认策略：FFT/频域模型优先 `use_amp=false`
2. 若必须开启 AMP，请手动设置 `use_amp=true` 并按提示调整 patch_size
3. 确保 patch_size 同时满足 scale 与模型整除要求

### Shape mismatch

**可能原因**：
- in_channels/out_channels 与数据变量数不匹配
- 模型参数配置中的 img_size 与实际数据不一致
- 模型输出空间尺寸与 HR 目标尺寸不一致（训练前会进行输出尺寸预检）

**修改建议**：
1. 确认 dyn_vars 数量与模型通道数一致
2. 检查 scale 和 HR shape 是否匹配
3. 对齐模型的 upsample_factor 或 patch_size，确保输出与 HR 尺寸一致

---

## 环境错误

### No module named 'torch'

**可能原因**：
- 使用了错误的 Python 路径（如 `/home/lz/miniconda3/bin/python`，base 环境无 torch）
- Agent 绕过训练工具手动拼接 bash 命令时使用了系统 Python

**修改建议**：
1. **必须通过 `ocean_sr_train_start` 工具启动训练**，工具内置 `findPythonWithModule('torch')` 自动选择正确环境
2. 若需手动验证：`/home/lz/miniconda3/envs/pytorch/bin/python -c "import torch; print(torch.__version__)"`
3. 参考 `references/command-templates.md` 获取正确的 Python 路径

### 其他 ModuleNotFoundError

**可能原因**：
- Python 环境中缺少依赖包（如 scipy、scikit-learn）
- 使用了错误的 conda 环境

**修改建议**：
1. 确认使用 pytorch 环境：`/home/lz/miniconda3/envs/pytorch/bin/python`
2. 安装缺失依赖：`/home/lz/miniconda3/envs/pytorch/bin/pip install <package>`

---

## DDP 错误

### unrecognized arguments: --local-rank

**可能原因**：
- 使用了废弃的 `torch.distributed.launch` launcher
- `torch.distributed.launch` 通过命令行参数传递 `--local-rank`，但新版 PyTorch 脚本使用环境变量 `LOCAL_RANK`

**修改建议**：
1. **禁止使用 `torch.distributed.launch`**，已废弃
2. 使用 `torch.distributed.run`（即 torchrun），通过环境变量传递 `LOCAL_RANK`
3. 训练工具已内置正确的 DDP launcher，直接使用工具即可

### RANK not set / KeyError: 'RANK'

**可能原因**：
- 直接运行 `python main_ddp.py` 而未通过 DDP launcher
- `main_ddp.py` 依赖 `RANK`/`LOCAL_RANK`/`WORLD_SIZE` 环境变量，这些由 `torch.distributed.run` 自动设置

**修改建议**：
1. **禁止直接运行 `main_ddp.py`**
2. 必须通过 `python -m torch.distributed.run --nproc_per_node=N main_ddp.py` 启动
3. 使用训练工具自动处理所有 DDP 配置

### Address already in use

**可能原因**：
- DDP master port（默认 29500）被其他训练进程占用
- 上一次训练进程未正确退出

**修改建议**：
1. 检查端口占用：`lsof -i :29500`
2. 终止占用进程，或使用训练工具（工具自动检测空闲端口 29500-29599）
3. 可通过 `ocean_sr_train_status({ action: "list" })` 检查残留进程

---

## Agent 操作错误

### Agent 手写 bash 训练命令失败

**症状**：Agent 绕过 `ocean_sr_train_start` 工具，手动拼接 bash 命令启动训练，连续多次失败。

**常见失败模式**：
1. 使用错误 Python 路径 → `No module named 'torch'`
2. 使用废弃 launcher → `--local-rank unrecognized`
3. 缺少环境变量 → `RANK not set`

**恢复流程**：
1. **立即停止手动尝试**
2. 回到训练工具：使用 `ocean_sr_train_start` 启动训练
3. 如果工具不满足需求，查阅 `references/command-templates.md` 获取正确命令模板
4. **规则**：训练启动必须使用训练工具，禁止手动拼接 bash 训练启动命令
