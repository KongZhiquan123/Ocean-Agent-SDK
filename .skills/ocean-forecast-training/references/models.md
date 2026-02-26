# 海洋时序预测模型参考

## 推荐模型排序

| 优先级 | 模型 | 类别 | 适用场景 |
|--------|------|------|----------|
| 1 | FNO2d | 频谱 | 默认首选，捕获全局模式，内存效率高 |
| 2 | UNet2d | CNN | 空间细节丰富，需 H/W 能被 16 整除 |
| 3 | SwinTransformerV2 | Transformer | 长程依赖建模，内存较高 |
| 4 | Transformer | Transformer | 通用 Transformer，小空间网格适用 |

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
3. **数据量大 (>500)**: SwinTransformerV2
4. **空间分辨率大 (>256x256)**: FNO2d 或 GalerkinTransformer
5. **空间分辨率小 (<64x64)**: Transformer 或 MLP
