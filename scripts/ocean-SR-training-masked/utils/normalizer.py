"""
normalizer.py - 归一化器（支持 NaN 和多通道数据）

@author Leizheng
@date 2026-02-06
@version 2.0.0

@changelog
  - 2026-02-06 Leizheng: v2.0.0 修复多通道广播 + torch.nanstd 兼容
    - 自定义 nanstd 替代不存在的 torch.nanstd（PyTorch 2.5.1）
    - UnitGaussianNormalizer: std 沿 dim=0 计算，形状 [N, C] 与 mean 一致
    - 修复 std 形状 [N] 导致多通道(C>1) encode 广播失败的 bug
  - 原始版本: v1.0.0 单通道版本
"""

import torch
from torch import nn


def nanstd(x, dim=None):
    """
    计算忽略 NaN 的标准差（兼容 PyTorch < 2.6 无 torch.nanstd 的情况）
    """
    valid_mask = ~torch.isnan(x)
    if dim is None:
        if valid_mask.any():
            return torch.std(x[valid_mask])
        else:
            return torch.tensor(1.0)
    else:
        x_filled = x.clone()
        x_filled[~valid_mask] = 0
        count = valid_mask.float().sum(dim=dim)
        count = count.clamp(min=1)
        mean = x_filled.sum(dim=dim) / count

        diff = x - mean.unsqueeze(dim)
        diff[~valid_mask] = 0
        var = (diff ** 2).sum(dim=dim) / count
        return torch.sqrt(var)


# normalization, pointwise gaussian
class UnitGaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x: [B, N, C] — B 个样本，N 个空间点，C 个通道
        # mean: [N, C], std: [N, C] — 沿 batch 维计算，与 x 广播兼容
        self.register_buffer('mean', torch.nanmean(x, 0))  # [N, C]
        self.register_buffer('std', nanstd(x, dim=0))       # [N, C]
        self.eps = eps

    def encode(self, x):
        # x: [B, N, C], mean: [N, C], std: [N, C] → 广播正确
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # [N, C]
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]
        B = x.shape[0]
        C = x.shape[-1]
        shape = x.shape
        x = x.view(B, -1, C)

        std = std.to(x.device)
        mean = mean.to(x.device)
        x = (x * std) + mean
        x = x.view(*shape)
        return x


# normalization, Gaussian (全局标量归一化，支持 NaN)
class GaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.nanmean(x)
        self.std = nanstd(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        B = x.shape[0]
        C = x.shape[-1]
        shape = x.shape
        x = x.view(B, -1, 1)
        x = (x * (self.std + self.eps)) + self.mean
        return x.view(*shape)
