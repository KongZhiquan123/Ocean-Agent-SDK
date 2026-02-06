import torch
from torch import nn


# normalization, pointwise gaussian
class UnitGaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        # 使用 nanmean 和 nanstd 处理 NaN
        self.register_buffer('mean', torch.nanmean(x, 0))
        self.register_buffer('std', torch.tensor([torch.nanstd(x[:, i]) for i in range(x.shape[1])]))
        self.eps = eps
    
    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]
        B = x.shape[0]
        C = x.shape[-1]
        shape = x.shape
        x = x.view(B, -1, C)
        
        # x is in shape of batch*n or T*batch*n
        std = std.to(x.device)
        mean = mean.to(x.device)
        x = (x * std) + mean
        x = x.view(*shape)
        return x


# normalization, Gaussian (修改版：支持 NaN)
class GaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        # 使用 nanmean 和 nanstd 忽略 NaN 值
        self.mean = torch.nanmean(x)
        # 计算标准差时也忽略 NaN
        valid_mask = ~torch.isnan(x)
        if valid_mask.any():
            valid_data = x[valid_mask]
            self.std = torch.std(valid_data)
        else:
            self.std = torch.tensor(1.0)
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
