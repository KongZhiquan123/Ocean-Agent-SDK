"""
ReMiG Trainer (masked version).

@author Leizheng
@date 2026-02-06
@version 2.0.0

@changelog
  - 2026-02-06 Leizheng: v2.0.0 loss 归一化用有效像素数（排除陆地）
  - 原始版本: v1.0.0
"""

from .base import BaseTrainer
from models import _ddpm_dict
from utils.loss import LossRecord


class ReMiGTrainer(BaseTrainer):
    def __init__(self, args):
        self.beta_schedule = args['beta_schedule']
        super().__init__(args)

    def build_model(self, **kwargs):
        model = _ddpm_dict[self.model_name]["model"](self.model_args)
        diffusion = _ddpm_dict[self.model_name]["diffusion"](
            model,
            model_args=self.model_args,
        )

        return diffusion

    def train(self, epoch, **kwargs):
        loss_record = LossRecord(["train_loss"])
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.model.train()
        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = x.permute(0, 3, 1, 2)
            y = y.permute(0, 3, 1, 2)
            data = {
                'SR': x,
                'HR': y
            }
            B, C, H, W = x.shape
            pix_loss = self.model(data)

            # 分母用有效像素数（排除陆地），而非全部像素
            if self.mask_hr is not None:
                valid_pixels_per_channel = self.mask_hr.sum().item()  # mask_hr: [1, H, W, 1]
                total_valid = B * C * valid_pixels_per_channel
                loss = pix_loss / max(total_valid, 1)
            else:
                loss = pix_loss / (B * C * H * W)

            loss_record.update({"train_loss": loss.item()}, n=B)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record

    def inference(self, x, y, **kwargs):
        x = x.permute(0, 3, 1, 2)
        y_pred = self._unwrap().super_resolution(x, continous=False)
        y_pred = y_pred.permute(0, 2, 3, 1).reshape(y.shape)
        return y_pred
