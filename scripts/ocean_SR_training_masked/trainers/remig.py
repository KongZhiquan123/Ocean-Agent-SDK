"""
ReMiG Trainer (masked version).

@author Leizheng
@date 2026-02-06
@version 3.3.0

@changelog
  - 2026-02-11 Leizheng: v3.3.0 inference 移除不兼容的 continous 参数
  - 2026-02-11 Leizheng: v3.2.0 修复 build_model 配置桥接
    - 从 args['resshift'] 读取 UNet/Diffusion 参数（与 ResshiftTrainer 一致）
    - 旧代码从 args['model'] 读取，缺少 model_channels 等架构参数导致 KeyError
  - 2026-02-10 Leizheng: v3.1.0 inference 改用动态 crop 替代脆弱的 reshape
  - 2026-02-07 Leizheng: v3.0.0 AMP 混合精度 + Gradient Checkpointing
    - train() 使用 autocast + GradScaler
    - gradient checkpointing 包装扩散模型 forward
  - 2026-02-06 Leizheng: v2.0.0 loss 归一化用有效像素数（排除陆地）
  - 原始版本: v1.0.0
"""

import torch
import torch.utils.checkpoint

from .base import BaseTrainer
from models import _ddpm_dict
from utils.loss import LossRecord
from utils.metrics import get_obj_from_str


class ReMiGTrainer(BaseTrainer):
    def __init__(self, args):
        self.beta_schedule = args['beta_schedule']
        super().__init__(args)

    def build_model(self, **kwargs):
        # ReMiG 架构参数在 resshift section 中（与 ResshiftTrainer 一致）
        self.resshift_cfg = self.args['resshift']

        # 构建 UNet（UNetModelSwin 接收 model_args dict，不是 **kwargs）
        model_params = self.resshift_cfg["model"]['params']
        model = get_obj_from_str(self.resshift_cfg['model']['target'])(model_params)

        # 构建 Diffusion（传入 model 和扩散参数）
        diffusion_params = self.resshift_cfg["diffusion"]['params']
        diffusion = _ddpm_dict[self.model_name]["diffusion"](
            model,
            model_args=diffusion_params,
        )

        return diffusion

    def train(self, epoch, **kwargs):
        loss_record = LossRecord(["train_loss"])
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            # Patch 训练时 batch = (x, y, mask_hr_patch)，否则 (x, y)
            if len(batch) == 3:
                x, y, mask_hr = batch
                mask_hr = mask_hr.to(self.device, non_blocking=True)
            else:
                x, y = batch
                mask_hr = self.mask_hr
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)
            B, C, H, W = x.shape

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if self.gradient_checkpointing:
                    def _forward(sr, hr):
                        return self.model({'SR': sr, 'HR': hr})
                    pix_loss = torch.utils.checkpoint.checkpoint(
                        _forward, x, y, use_reentrant=False
                    )
                else:
                    pix_loss = self.model({'SR': x, 'HR': y})

                # 分母用有效像素数（排除陆地），而非全部像素
                if mask_hr is not None:
                    mask_expanded = mask_hr.expand(B, -1, -1, -1)
                    total_valid = mask_expanded.sum().item() * C
                    loss = pix_loss / max(total_valid, 1)
                else:
                    loss = pix_loss / (B * C * H * W)

            loss_record.update({"train_loss": loss.item()}, n=B)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record

    def inference(self, x, y, **kwargs):
        x = x.permute(0, 3, 1, 2)
        y_pred = self._unwrap().super_resolution(x)
        y_pred = y_pred.permute(0, 2, 3, 1)  # [B, C, H', W'] -> [B, H', W', C]
        # 动态 crop 到 y 的空间尺寸（alignment padding 安全网）
        y_pred = y_pred[:, :y.shape[1], :y.shape[2], :]
        return y_pred
