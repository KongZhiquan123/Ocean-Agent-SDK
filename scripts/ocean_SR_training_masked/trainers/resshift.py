"""
ResShift Trainer (masked version).

@author Leizheng
@contributors Leizheng
@date 2026-02-06
@version 4.0.0

@changelog
  - 2026-02-07 Leizheng: v4.0.0 inference 自动对齐尺寸 + crop 回原尺寸
    - interpolate 到 model_divisor 对齐后的尺寸
    - 采样完成后 crop 回原始 y 尺寸
  - 2026-02-07 Leizheng: v3.0.0 AMP 混合精度 + Gradient Checkpointing
    - train() 使用 autocast + GradScaler
    - gradient checkpointing 包装 training_losses forward
  - 2026-02-06 Leizheng: v2.0.0 masked 版本
    - 训练 loss 来自 base_diffusion.training_losses()，无法直接注入 mask
    - 评估阶段通过继承 BaseTrainer.evaluate() 使用 masked metrics
  - 原始版本: v1.0.0
"""

import torch
import torch.distributed as dist
import torch.utils.checkpoint
import functools
import numpy as np
from models import _ddpm_dict
from utils.loss import LossRecord
import torch.nn.functional as F
from .base import BaseTrainer

from utils.metrics import get_obj_from_str

class ResshiftTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def build_model(self, **kwargs):

        self.resshift_cfg = self.args['resshift']

        params = self.resshift_cfg["model"]['params']
        model =get_obj_from_str(self.resshift_cfg['model']['target'])(**params)

        params = self.resshift_cfg["diffusion"]['params']
        self.base_diffusion = get_obj_from_str(self.resshift_cfg['diffusion']['target'])(**params)
        return model

    def train(self, epoch, **kwargs):
        loss_record = LossRecord(["train_loss"])
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            # Patch 训练时 batch = (x, y, mask_hr_patch)，否则 (x, y)
            if len(batch) == 3:
                x, y, _mask_hr = batch
            else:
                x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)
            x = F.interpolate(x, size=y.shape[2:], mode='bicubic', align_corners=False)

            B, C, H, W = x.shape

            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(y.shape[0],),
                    device=x.device,
                    )

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if self.gradient_checkpointing:
                    def _forward(gt, lq, timesteps):
                        model_kwargs = {'lq': lq}
                        losses, z0, zt = self.base_diffusion.training_losses(
                            self.model, gt, lq, timesteps,
                            first_stage_model=None, model_kwargs=model_kwargs, noise=None,
                        )
                        return losses['mse'], z0, zt
                    loss, z0_pred, z_t = torch.utils.checkpoint.checkpoint(
                        _forward, y, x, tt, use_reentrant=False
                    )
                else:
                    model_kwargs = {'lq': x}
                    compute_losses = functools.partial(
                        self.base_diffusion.training_losses,
                        self.model, y, x, tt,
                        first_stage_model=None,
                        model_kwargs=model_kwargs,
                        noise=None,
                    )
                    losses, z0_pred, z_t = compute_losses()
                    loss = losses['mse']

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
        y = y.permute(0, 3, 1, 2)

        orig_h, orig_w = y.shape[2], y.shape[3]

        # interpolate 到对齐后的尺寸（能被 model_divisor 整除）
        d = self.model_divisor
        if d > 1:
            aligned_h = ((orig_h + d - 1) // d) * d
            aligned_w = ((orig_w + d - 1) // d) * d
        else:
            aligned_h, aligned_w = orig_h, orig_w
        x = F.interpolate(x, size=(aligned_h, aligned_w), mode='bicubic', align_corners=False)

        B, C, H, W = x.shape

        tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(y.shape[0],),
                    device=x.device,
                    )

        indices = np.linspace(
                    0,
                    self.base_diffusion.num_timesteps,
                    self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()

        if not (self.base_diffusion.num_timesteps-1) in indices:
            indices.append(self.base_diffusion.num_timesteps-1)

        im_lq = x

        model_kwargs = {'lq':x,}

        tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        device= x.device
                        )

        self._unwrap()

        y_pred = self.base_diffusion.p_sample_loop(
                        y=im_lq,
                        model=self.model,
                        first_stage_model=None,
                        noise=None,
                        clip_denoised=None,
                        model_kwargs=model_kwargs,
                        device=x.device,
                        progress=True,
                        )

        # crop 回原始尺寸
        y_pred = y_pred[:, :, :orig_h, :orig_w]
        y_pred = y_pred.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        return y_pred
