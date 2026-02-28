# 海洋时序预测模型参考

## 推荐模型排序

| 优先级 | 模型 | 类别 | 适用场景 |
|--------|------|------|----------|
| 1 | FNO2d | 频谱 | 默认首选，捕获全局模式，内存效率高 |
| 2 | UNet2d | CNN | 空间细节丰富，需 H/W 能被 16 整除 |
| 3 | SwinTransformerV2 | Transformer | 长程依赖建模，内存较高 |
| 4 | Fuxi | Transformer | 气象预测架构，Swin-based 2D/3D 双路径，大网格适用 |
| 5 | Crossformer | Transformer | 时空两阶段注意力，显式时间建模，多步预测适用 |
| 6 | Transformer | Transformer | 通用 Transformer，小空间网格适用 |

## 全部模型列表

### FNO2d（Fourier Neural Operator 2D）
- **类别**: 频谱方法
- **特点**: 通过 FFT 在频域做卷积，捕获全局空间模式
- **推荐配置**:
  - modes: [15, 12, 9, 9, 9]
  - width: 64
  - act: gelu
- **内存估算**: ~2GB (batch=4, 128x128, in_t=7, 2 vars)
- **注意**: AMP 默认关闭（FFT 对半精度敏感）

### UNet2d
- **类别**: CNN
- **特点**: 编码器-解码器架构，擅长捕获多尺度空间特征
- **要求**: 空间尺寸 H, W 必须能被 16 整除
- **内存估算**: ~3GB (batch=4, 128x128)
- **推荐场景**: 空间分辨率适中，需要精细空间细节

### SwinTransformerV2
- **类别**: Transformer
- **特点**: 移位窗口注意力机制，高效长程依赖建模
- **内存估算**: ~5GB (batch=4, 128x128)
- **注意**: 属于 HEAVY_MODELS，默认开启 gradient_checkpointing

### Transformer
- **类别**: Transformer
- **特点**: 标准 Transformer + 位置编码
- **限制**: O(N^2) 内存，大空间网格不适用
- **推荐场景**: 空间网格 < 64x64

### M2NO2d
- **类别**: 频谱方法
- **特点**: 多分辨率神经算子
- **注意**: AMP 默认关闭

### GalerkinTransformer
- **类别**: Transformer
- **特点**: 线性复杂度注意力
- **推荐场景**: 大空间网格

### NeuralFramework 模型（气象预测架构改编）

#### OceanCNN
- **类别**: CNN（基线）
- **特点**: 简单编码器-解码器 CNN，3 层下采样 + 3 层上采样
- **要求**: H, W 必须能被 8 整除
- **推荐场景**: 快速基线测试

#### OceanResNet
- **类别**: CNN（基线）
- **特点**: ResNet-18 骨干 + 转置卷积解码器，自动空间对齐
- **推荐场景**: 快速基线，比 OceanCNN 更强

#### OceanViT
- **类别**: Transformer
- **特点**: 标准 Vision Transformer，patch 级别处理
- **配置**: patch_size=8, d_model=256, nhead=8, num_layers=6
- **要求**: H, W 必须能被 patch_size 整除

#### Fuxi ⭐
- **类别**: Transformer
- **特点**: Swin-based 2D/3D 双路径处理，UTransformer 架构
- **配置**: embed_dim=192, num_heads=6, window_size=7, depth=8
- **依赖**: timm
- **内存**: 较高（属于 HEAVY_MODELS）
- **推荐场景**: 大空间网格，需要强全局建模能力

#### Fengwu
- **类别**: Transformer
- **特点**: 多尺度 2D+3D 编码器-解码器，Earth-aware 注意力
- **配置**: embed_dim=192, num_heads=6, window_size=[6,6], depth=6
- **依赖**: einops
- **内存**: 高（属于 HEAVY_MODELS）

#### Pangu
- **类别**: Transformer
- **特点**: 地球感知位置偏置 + 2D/3D 混合路径
- **配置**: embed_dim=192, num_heads=6, window_size=[6,6], depth=6
- **依赖**: einops
- **内存**: 高（属于 HEAVY_MODELS）

#### Crossformer ⭐
- **类别**: Transformer
- **特点**: 时间-空间两阶段注意力（TSA），显式时间建模
- **配置**: d_model=256, n_heads=4, seg_len=6, e_layers=3
- **推荐场景**: 多步预测，需要显式时间维度建模

#### NNG
- **类别**: 图神经网络
- **特点**: 基于正二十面体网格的消息传递，纯 PyTorch 实现
- **依赖**: scikit-learn, scipy
- **内存**: 高（属于 HEAVY_MODELS）
- **配置**: hidden_dim=256, num_processor_layers=8, mesh_level=3

#### OneForecast
- **类别**: 图神经网络
- **特点**: 简化图操作，纯 PyTorch index_add_ 消息传递，DataParallel 兼容
- **依赖**: scikit-learn, scipy
- **配置**: hidden_dim=256, num_processor_layers=8, mesh_level=3

#### GraphCast
- **类别**: 图神经网络
- **特点**: 网格级消息传递 + 可选 RNN 时间编码，纯 PyTorch 实现
- **依赖**: scikit-learn, scipy
- **内存**: 高（属于 HEAVY_MODELS）
- **配置**: hidden_dim=256, num_processor_layers=8, mesh_level=3

### 其他模型
- **SwinMLP**: MLP 版本的 Swin 架构
- **Transolver**: 物理感知 Transformer
- **GNOT**: 通用神经算子 Transformer
- **ONO**: 正交神经算子
- **LSM**: 潜在频谱模型
- **LNO**: Laplace 神经算子
- **MLP**: 简单多层感知机（基线）
- **UNet1d/FNO1d**: 1D 版本（适用于 1D 空间域）
- **UNet3d/FNO3d**: 3D 版本（体积数据）

## 模型选择建议

1. **数据量少 (<100 时间步)**: FNO2d（参数少，不易过拟合）
2. **数据量中等 (100-500)**: UNet2d 或 FNO2d
3. **数据量大 (>500)**: SwinTransformerV2 或 Fuxi
4. **空间分辨率大 (>256x256)**: FNO2d 或 GalerkinTransformer
5. **空间分辨率小 (<64x64)**: Transformer 或 MLP
6. **多步预测**: Crossformer（显式时间建模优势）
7. **气象预测风格**: Fuxi > Fengwu > Pangu（从简到复杂）
8. **图方法探索**: OneForecast > NNG > GraphCast（需安装 scikit-learn + scipy）
