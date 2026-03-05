# 训练故障排查指南（超分辨率）

> 版本: 1.0.0 | 最后更新: 2026-03-04

---

## 错误诊断决策树

遇到训练问题时，按以下分支逐步排查：

### 分支 A：启动崩溃（进程启动后立即退出）

```
进程立即退出
├── 错误含 "No module named 'torch'"
│   → 根因：使用了错误的 Python 路径（如 miniconda3/bin/python 无 torch）
│   → 方案：必须通过训练工具启动，工具会自动选择含 torch 的 Python
│   → 参考：command-templates.md #Python 环境
│
├── 错误含 "--local-rank" / "unrecognized arguments"
│   → 根因：使用了废弃的 torch.distributed.launch launcher
│   → 方案：使用 torch.distributed.run（torchrun），工具已内置正确 launcher
│   → 参考：command-templates.md #禁止命令
│
├── 错误含 "RANK" / "LOCAL_RANK" not set / KeyError
│   → 根因：直接运行 main_ddp.py 而未通过 DDP launcher
│   → 方案：DDP 训练必须通过训练工具启动，工具自动设置环境变量
│   → 参考：command-templates.md #DDP 启动模板
│
├── 错误含 "Address already in use"
│   → 根因：DDP master port 被占用
│   → 方案：工具自动检测空闲端口（29500-29599），或先终止占用进程
│   → 参考：command-templates.md #调试命令
│
└── 错误含 "FileNotFoundError" / "No such file"
    → 根因：数据路径不正确或 best_model.pth 不存在
    → 方案：检查 dataset_root 和 log_dir 路径
```

### 分支 B：训练中途失败（已开始训练但异常退出）

```
训练中途失败
├── 错误含 "CUDA out of memory"
│   → 根因：GPU 显存不足
│   → 方案：减小 batch_size / 开启 AMP / 开启 gradient_checkpointing / 设置 patch_size
│   → 参考：errors.md #CUDA out of memory
│
├── 错误含 "NCCL error" / "NCCL timeout"
│   → 根因：GPU 间通信失败或节点挂起
│   → 方案：检查 GPU 可用性，尝试减少 GPU 数量或切换 DP 模式
│
├── 错误含 "NaN" / "loss is nan"
│   → 根因：学习率过大 / FFT+AMP 不兼容 / 数据异常
│   → 方案：降低 lr / FFT 模型关闭 AMP / 检查数据
│
├── 错误含 "CUDA device-side assert"
│   → 根因：张量索引越界或数据标签超出范围
│   → 方案：设置 CUDA_LAUNCH_BLOCKING=1 获取详细堆栈，检查数据完整性
│
└── 错误含 "Shape mismatch" / "size mismatch"
    → 根因：模型输入输出通道与数据不匹配 / ckpt 与当前模型架构不兼容
    → 方案：确认 dyn_vars / scale 正确；续训时不可更换模型架构
```

### 分支 C：Agent bash 失败（Agent 绕过工具手动执行）

```
Agent 手动执行 bash 训练命令失败
├── 使用了错误的 Python 路径
│   → 恢复：停止手动尝试，改用 ocean_sr_train_start 工具
│
├── 使用了废弃的 launcher
│   → 恢复：停止手动尝试，改用 ocean_sr_train_start 工具
│
└── 缺少环境变量（RANK / CUDA_VISIBLE_DEVICES 等）
    → 恢复：停止手动尝试，改用 ocean_sr_train_start 工具
    → 规则：训练启动必须使用训练工具，禁止手动拼接 bash 命令
```

---

## 常见错误速查表

| # | 错误特征 | 根因 | 解决方案 |
|---|---------|------|---------|
| 1 | `No module named 'torch'` | Python 路径错误，使用了无 torch 的环境 | 通过训练工具启动（工具内置 `findPythonWithModule('torch')` 自动选择） |
| 2 | `unrecognized arguments: --local-rank` | 使用废弃的 `torch.distributed.launch` | 工具已使用 `torch.distributed.run`，禁止手动拼 launch 命令 |
| 3 | `RANK not set` / `KeyError: 'RANK'` | 直接运行 `main_ddp.py` 未通过 DDP launcher | 必须通过训练工具启动 DDP，工具自动设置 RANK/LOCAL_RANK |
| 4 | `Address already in use (port 29500)` | DDP master port 被占用 | 工具自动检测空闲端口；手动排查：`lsof -i :29500` |
| 5 | `NCCL error` / `NCCL timeout` | GPU 间通信失败 | 检查 GPU 可用性、减少 GPU 数量、尝试 DP 模式 |
| 6 | `Shape mismatch` / `size mismatch` | 模型通道数与数据不匹配，或 ckpt 跨模型加载 | 确认 dyn_vars 数量正确；续训不可更换模型架构 |
| 7 | `best_model.pth not found` | 续训或 predict 时 ckpt 路径不存在 | 确认 log_dir 下有 best_model.pth，或显式传入 ckpt_path |
| 8 | `CUDA device-side assert` | 张量索引越界 / 数据标签异常 | 设置 `CUDA_LAUNCH_BLOCKING=1` 获取详细堆栈 |
| 9 | `cuFFT doesn't support signals of half type` | FFT 模型开启了 AMP（FP16） | FFT/频域模型设置 `use_amp=false` |
| 10 | `RuntimeError: CUDA out of memory` | GPU 显存不足 | 按优先级：AMP → 减 batch_size → gradient_checkpointing → patch_size |

---

## 工作空间文件结构

训练工具会将 `scripts/ocean-sr-training/` 复制到工作空间 `{log_dir}/_ocean_sr_code/`。

### 目录树

```
{log_dir}/
├── _ocean_sr_code/            ← 工作空间副本（可修改）
│   ├── main.py                ← 单卡/DP 入口
│   ├── main_ddp.py            ← DDP 入口
│   ├── config.py              ← 配置解析
│   ├── utils/                 ← 工具函数
│   ├── models/                ← 仅包含选中模型 + base/
│   │   ├── __init__.py        ← 自动生成的模型注册表
│   │   ├── base/              ← 共享基础模块
│   │   └── {model_name}/      ← 选中的模型代码
│   ├── trainers/              ← 训练器
│   └── datasets/              ← 数据集加载
├── config.yaml                ← 训练配置
├── train.log                  ← 训练日志
├── best_model.pth             ← 最佳模型权重
└── plots/                     ← 可视化输出
```

### 修改权限

| 目录/文件 | 可修改 | 说明 |
|-----------|--------|------|
| `{log_dir}/_ocean_sr_code/**` | 是 | 工作空间副本，可自由修改调试 |
| `scripts/ocean-sr-training/**` | **否** | 原始目录，禁止修改 |
| `{log_dir}/config.yaml` | 是 | 训练配置，可查看/修改后重启 |
| `{log_dir}/*.log` | 只读 | 日志文件，仅供查看 |
