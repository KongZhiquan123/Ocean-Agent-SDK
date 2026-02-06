# 模型详细说明

> 版本: 1.0.0 | 最后更新: 2026-02-06

---

## 标准模型（BaseTrainer）

### FNO2d - Fourier Neural Operator 2D
- **特点**：基于傅里叶变换的算子学习，擅长捕获全局频率特征
- **适用场景**：物理场数据（流速、温度等）
- **关键参数**：`modes1`, `modes2`（傅里叶模态数）, `width`（通道宽度）

### UNet2d
- **特点**：经典编码-解码结构，通用性强
- **适用场景**：通用超分任务

### M2NO2d - Multiplicative Multiresolution Neural Operator
- **特点**：多分辨率算子，层次化特征提取
- **适用场景**：多尺度物理场

### SwinIR（推荐）
- **特点**：基于 Swin Transformer 的图像超分辨率，效果优秀
- **适用场景**：通用超分任务，海洋数据效果好
- **关键参数**：`embed_dim`, `depths`, `num_heads`, `window_size`

### EDSR - Enhanced Deep Super-Resolution
- **特点**：深度残差网络，结构简洁高效
- **适用场景**：快速训练验证

### HiNOTE - High-order Neural Operator
- **特点**：高阶神经算子
- **适用场景**：需要高阶精度的物理场

### Galerkin_Transformer
- **特点**：基于 Galerkin 方法的 Transformer
- **适用场景**：PDE 求解相关任务

### MWT2d - Morlet Wavelet Transform
- **特点**：基于小波变换
- **适用场景**：多尺度分析

### SRNO - Super-Resolution Neural Operator
- **特点**：专用于超分辨率的神经算子
- **适用场景**：物理场超分

### Swin_Transformer
- **特点**：原版 Swin Transformer 超分
- **适用场景**：通用超分

---

## 扩散模型

### DDPM - Denoising Diffusion Probabilistic Model
- **Trainer**：DDPMTrainer
- **特点**：扩散去噪模型，生成质量高但训练较慢
- **关键参数**：`n_timestep`（扩散步数）, `beta_schedule`
- **注意**：训练时间显著长于标准模型

### SR3 - Super-Resolution via Repeated Refinement
- **Trainer**：DDPMTrainer
- **特点**：条件扩散模型，专为超分设计
- **适用场景**：高质量超分结果

### MG-DDPM - Multigrid DDPM
- **Trainer**：DDPMTrainer
- **特点**：多网格加速的扩散模型
- **适用场景**：加速扩散训练

### ReMiG
- **Trainer**：DDPMTrainer
- **特点**：基于 Swin Transformer 的扩散模型
- **适用场景**：高质量超分

### Resshift
- **Trainer**：ResshiftTrainer
- **特点**：残差偏移扩散模型
- **适用场景**：高效扩散训练
