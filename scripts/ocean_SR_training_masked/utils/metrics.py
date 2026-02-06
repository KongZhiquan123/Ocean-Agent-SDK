"""
Evaluation metrics for ocean SR training (masked version).

@author Leizheng
@date 2026-02-06
@version 2.0.0

@changelog
  - 2026-02-06 Leizheng: v2.0.0 添加 masked 评估指标
    - 新增 masked_mse, masked_rmse, masked_psnr, masked_ssim
    - 新增 MaskedEvaluator，所有指标只在海洋格点上计算
  - 原始版本: v1.0.0
"""

import torch
import torch.nn.functional as F

from typing import Dict, List, Any
from .loss import LossRecord

import importlib


# ========================================
# 原始指标（保留兼容）
# ========================================

@torch.no_grad()
def mse(pred, target, *args, **kwargs):
    return F.mse_loss(pred, target, reduction="mean")


@torch.no_grad()
def rmse(pred, target, *args, **kwargs):
    return torch.sqrt(mse(pred, target) + 1e-12)


@torch.no_grad()
def psnr(pred, target, shape, data_range=None, eps=1e-12):
    m = mse(pred, target)
    pred = pred.permute(0, 3, 1, 2)  # BCHW
    target = target.permute(0, 3, 1, 2)  # BCHW

    L = (target.max() - target.min()).clamp_min(eps)

    return 20.0 * torch.log10(L) - 10.0 * torch.log10(m + eps)


@torch.no_grad()
def ssim(pred, target, shape, data_range=None, K1=0.01, K2=0.03, eps=1e-12):
    """
    仅自适配 C1/C2：
      C1 = (K1*L)^2, C2 = (K2*L)^2, 其中 L=data_range
    其他计算保持你原版不变（3x3 平均池化）
    """
    pred = pred.permute(0, 3, 1, 2)  # BCHW
    target = target.permute(0, 3, 1, 2)  # BCHW

    # 自适配 L
    if data_range is None:
        L = (target.max() - target.min()).clamp_min(eps)
    else:
        L = torch.as_tensor(data_range, device=pred.device, dtype=pred.dtype).clamp_min(eps)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2) + eps
    )
    return ssim_map.mean()


METRIC_REGISTRY = {
    "mse": mse,
    "rmse": rmse,
    "psnr": psnr,
    "ssim": ssim,
}


class Evaluator:
    def __init__(self, shape: List[int], **metric_kwargs: Any):
        self.kw = metric_kwargs
        self.shape = shape

    def init_record(self, loss_list: List[str] = []) -> LossRecord:
        loss_list = loss_list + list(METRIC_REGISTRY.keys())
        return LossRecord(loss_list)

    @torch.no_grad()
    def __call__(self, pred, target, record=None, batch_size=None, **batch) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, fn in METRIC_REGISTRY.items():
            out[name] = fn(pred, target, self.shape, **self.kw).item()
        if record is not None:
            record.update(out)

        return out


# ========================================
# Masked 指标（排除陆地格点）
# ========================================

@torch.no_grad()
def masked_mse(pred, target, shape, mask=None, **kwargs):
    """
    只在海洋格点上计算 MSE，分母 = 海洋格点数。

    Args:
        pred, target: [B, H, W, C]
        mask: [1, H, W, 1] bool，True=海洋
    """
    if mask is None:
        return mse(pred, target)

    mask_expanded = mask.expand_as(pred).float()  # [B, H, W, C]
    valid_count = mask_expanded.sum().clamp(min=1.0)
    diff_sq = (pred - target) ** 2
    return (diff_sq * mask_expanded).sum() / valid_count


@torch.no_grad()
def masked_rmse(pred, target, shape, mask=None, **kwargs):
    return torch.sqrt(masked_mse(pred, target, shape, mask=mask) + 1e-12)


@torch.no_grad()
def masked_psnr(pred, target, shape, mask=None, eps=1e-12, **kwargs):
    """基于 masked_mse 计算 PSNR"""
    m = masked_mse(pred, target, shape, mask=mask)

    if mask is not None:
        # 只在海洋格点上计算 data range
        mask_expanded = mask.expand_as(target)
        ocean_values = target[mask_expanded]
        L = (ocean_values.max() - ocean_values.min()).clamp_min(eps)
    else:
        L = (target.max() - target.min()).clamp_min(eps)

    return 20.0 * torch.log10(L) - 10.0 * torch.log10(m + eps)


@torch.no_grad()
def masked_ssim(pred, target, shape, mask=None, K1=0.01, K2=0.03, eps=1e-12, **kwargs):
    """
    在海洋区域计算 SSIM。
    策略：将陆地区域填充为 0，计算 SSIM map，然后只在海洋像素上取平均。
    """
    if mask is None:
        return ssim(pred, target, shape)

    pred_bchw = pred.permute(0, 3, 1, 2)  # BCHW
    target_bchw = target.permute(0, 3, 1, 2)  # BCHW

    # mask: [1, H, W, 1] → [1, 1, H, W] for BCHW format
    mask_bchw = mask.permute(0, 3, 1, 2).float()  # [1, 1, H, W]

    # 将陆地区域填充为 0
    pred_masked = pred_bchw * mask_bchw.expand_as(pred_bchw)
    target_masked = target_bchw * mask_bchw.expand_as(target_bchw)

    # 只在海洋格点上计算 data range
    mask_expanded = mask.expand_as(target)
    ocean_values = target[mask_expanded]
    L = (ocean_values.max() - ocean_values.min()).clamp_min(eps)

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_x = F.avg_pool2d(pred_masked, 3, 1, 1)
    mu_y = F.avg_pool2d(target_masked, 3, 1, 1)
    sigma_x = F.avg_pool2d(pred_masked * pred_masked, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(target_masked * target_masked, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(pred_masked * target_masked, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2) + eps
    )

    # 只在海洋区域取平均
    mask_bchw_expanded = mask_bchw.expand_as(ssim_map)
    valid_count = mask_bchw_expanded.sum().clamp(min=1.0)
    return (ssim_map * mask_bchw_expanded).sum() / valid_count


MASKED_METRIC_REGISTRY = {
    "mse": masked_mse,
    "rmse": masked_rmse,
    "psnr": masked_psnr,
    "ssim": masked_ssim,
}


class MaskedEvaluator(Evaluator):
    """
    使用 masked 指标的评估器。所有指标只在海洋格点上计算。
    """
    def init_record(self, loss_list: List[str] = []) -> LossRecord:
        loss_list = loss_list + list(MASKED_METRIC_REGISTRY.keys())
        return LossRecord(loss_list)

    @torch.no_grad()
    def __call__(self, pred, target, record=None, mask=None, **kwargs) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, fn in MASKED_METRIC_REGISTRY.items():
            out[name] = fn(pred, target, self.shape, mask=mask, **self.kw).item()
        if record is not None:
            record.update(out)
        return out


# ========================================
# 工具函数
# ========================================

#用于resshift的模型构建函数：
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
