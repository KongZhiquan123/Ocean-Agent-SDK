# 对话示例

> 版本: 1.0.0 | 最后更新: 2026-02-06

---

## 示例 1：完整流程

```
用户: 我的预处理数据在 /data/output/demo14，帮我训练一个超分模型，输出到 /data/logs

Agent: 好的，我先确认一下您的数据目录...

[检查目录结构]

数据目录确认：
- 训练集: 255 个时间步
- 验证集: 55 个时间步
- 测试集: 55 个时间步
- 变量: temp
- HR shape: [680, 1440]
- LR shape: [170, 360]

接下来请选择模型。以下是可用的超分模型：

【标准模型】
- FNO2d: Fourier Neural Operator
- SwinIR: SwinIR 超分辨率（推荐）
- EDSR: Enhanced Deep Super-Resolution
- ...

【扩散模型】
- DDPM: 扩散去噪模型
- SR3: 超分专用扩散模型
- ...

您想使用哪个模型？

用户: SwinIR

Agent: 好的，选择 SwinIR。现在确认训练参数：

[查看 GPU]

当前可用 GPU：
- GPU 0: RTX 4090 (23.6GB, 空闲 23.3GB)
- GPU 1: RTX 4090 (23.6GB, 空闲 23.3GB)
- ...共 8 张

请确认以下参数：
1. 训练轮数 (epochs)？默认 500
2. 学习率 (lr)？默认 0.001
3. Batch size？默认 32
4. 使用哪些 GPU？（可选单卡或多卡）

用户: 500 epochs，lr 0.001，batch 64，用 0 和 1 两张卡

Agent: 收到。执行前请确认所有参数：

【训练参数确认】
- 数据目录: /data/output/demo14
- 输出目录: /data/logs
- 模型: SwinIR
- 变量: temp
- 超分倍数: 4x
- Epochs: 500
- 学习率: 0.001
- Batch size: 64
- GPU: 2 张 (GPU 0, GPU 1)，DDP 模式
- 归一化: PGN

确认执行？

用户: 确认

Agent: [开始训练...]
```

---

## 示例 2：单卡训练

```
用户: 数据在 /output/ocean，用 FNO2d 训练，单卡 GPU 0，100 epochs 测试一下

Agent: 好的，我先确认数据目录和 GPU...

[确认数据 + 查看 GPU]

参数确认：
- 数据: /output/ocean
- 模型: FNO2d
- Epochs: 100
- GPU: 单卡 (GPU 0)
- 其余使用默认值

确认执行？

用户: 确认

Agent: [开始训练...]
```

---

## 示例 3：错误处理

```
Agent: [执行训练...]

训练出错：

错误信息：
  CUDA out of memory. Tried to allocate 2.50 GB

可能原因：
- Batch size 过大，单卡显存不足
- 模型参数量大（如扩散模型），GPU 显存不够

修改建议：
1. 减小 batch_size（当前 64 → 建议 16 或 32）
2. 使用多卡训练分摊显存
3. 选择更轻量的模型（如 EDSR、FNO2d）

请问您希望如何调整？
```
