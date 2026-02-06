"""
Loss functions for ocean SR training (masked version).

@author Leizheng
@date 2026-02-06
@version 2.0.0

@changelog
  - 2026-02-06 Leizheng: v2.0.0 添加 MaskedLpLoss
    - 支持显式 mask 参数，只在海洋格点上计算 loss
    - 求平均时分母 = 海洋格点数（排除陆地格点）
  - 原始版本: v1.0.0
"""

import torch
import torch.nn.functional as F
from time import time
import torch.distributed as dist

_loss_dict = {

}


class CompositeLoss:
    """
    组合损失：传入 {name: weight}，自动求和并返回总损失与分项日志
    """
    def __init__(self, spec: dict[str, float]):  # e.g. {"l1":1.0,"l2":0.1,"physics":0.5}
        self.loss_list = ["total_loss", "l2", "l1"]
        self.init_record()

    def __call__(self, pred, target, *, batch_size: int | None = None, **batch):
        logs = {}
        total = 0.0
        for name, w in self.spec.items():
            if w == 0: 
                continue
            fn = LOSS_REGISTRY[name]
            val = fn(pred, target, **batch)  # 标量（已mean）
            total = total + w * val
            logs[name] = float(val.detach().item())
        logs["loss_total"] = float(total.detach().item())
        if record is not None and batch_size is not None:
            record.update(logs, n=batch_size)
        return total  # 用于 backward

    def init_record(self):
        self.record = LossRecord(self.loss_list)

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    支持 NaN 掩码 - 修改版
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        
        # 创建掩码：标记有效值（非 NaN）
        mask = ~torch.isnan(y)
        
        # 如果全是 NaN，返回 0
        if not mask.any():
            return torch.tensor(0.0, device=x.device)
        
        # 将 NaN 替换为 0（不影响计算，因为会被掩码过滤）
        x_masked = torch.where(mask, x, torch.zeros_like(x))
        y_masked = torch.where(mask, y, torch.zeros_like(y))
        
        # 展平并只保留有效值
        x_flat = x_masked.reshape(num_examples, -1)
        y_flat = y_masked.reshape(num_examples, -1)
        mask_flat = mask.reshape(num_examples, -1)
        
        # 对每个样本计算相对误差
        diff_norms = []
        y_norms = []
        
        for i in range(num_examples):
            valid_mask = mask_flat[i]
            if valid_mask.sum() == 0:
                # 如果该样本全是 NaN，跳过
                continue
            
            x_valid = x_flat[i][valid_mask]
            y_valid = y_flat[i][valid_mask]
            
            diff_norm = torch.norm(x_valid - y_valid, self.p)
            y_norm = torch.norm(y_valid, self.p)
            
            diff_norms.append(diff_norm)
            y_norms.append(y_norm)
        
        if len(diff_norms) == 0:
            return torch.tensor(0.0, device=x.device)
        
        diff_norms = torch.stack(diff_norms)
        y_norms = torch.stack(y_norms)
        
        # 避免除零
        y_norms = torch.clamp(y_norms, min=1e-8)
        
        rel_errors = diff_norms / y_norms
        
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_errors)
            else:
                return torch.sum(rel_errors)
        
        return rel_errors

    def __call__(self, x, y, **kwargs):
        return self.rel(x, y)


class MaskedLpLoss(object):
    """
    带显式 mask 的 Lp Loss。
    求平均时分母只算海洋格点数（不算陆地格点）。

    与 LpLoss 的区别：
    - LpLoss 通过检测 NaN 来推断 mask
    - MaskedLpLoss 接受显式 mask 参数（数据中 NaN 已被填充为 0）
    """
    def __init__(self, p=2, reduction=True, size_average=True):
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y, mask=None, **kwargs):
        """
        x, y: [B, H, W, C]
        mask: [1, H, W, 1] bool，True=海洋（有效像素），False=陆地
              如果 mask=None，退化为在所有像素上计算（等价于原 LpLoss 无 NaN 场景）
        """
        B = x.size(0)
        x_flat = x.reshape(B, -1)
        y_flat = y.reshape(B, -1)

        if mask is not None:
            mask_flat = mask.expand_as(x).reshape(B, -1).float()
        else:
            mask_flat = torch.ones_like(x_flat)

        # 只在海洋格点上计算差异
        diff = (x_flat - y_flat) * mask_flat
        y_masked = y_flat * mask_flat

        diff_norms = torch.norm(diff, self.p, dim=1)
        y_norms = torch.norm(y_masked, self.p, dim=1).clamp(min=1e-8)
        rel_errors = diff_norms / y_norms

        if self.reduction:
            if self.size_average:
                return rel_errors.mean()
            else:
                return rel_errors.sum()
        return rel_errors


class AverageRecord(object):
    """Computes and stores the average and current values for multidimensional data"""

    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class LossRecord:
    """
    A class for keeping track of loss values during training.

    Attributes:
        start_time (float): The time when the LossRecord was created.
        loss_list (list): A list of loss names to track.
        loss_dict (dict): A dictionary mapping each loss name to an AverageRecord object.
    """

    def __init__(self, loss_list):
        self.start_time = time()
        self.loss_list = loss_list
        self.loss_dict = {loss: AverageRecord() for loss in self.loss_list}
    
    def update(self, update_dict, n=1):
        for key, value in update_dict.items():
            self.loss_dict[key].update(value, n)
    
    def format_metrics(self):
        result = ""
        for loss in self.loss_list:
            result += "{}: {:.8f} | ".format(loss, self.loss_dict[loss].avg)
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result
    
    def to_dict(self):
        return {
            loss: self.loss_dict[loss].avg for loss in self.loss_list
        }
        
    def dist_reduce(self, device=None):
        if not (dist.is_available() and dist.is_initialized()):
            return

        device = device if device is not None else (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available() else torch.device("cpu")
        )

        for loss in self.loss_list:
            # 打包 sum 与 count，一次 all_reduce 两次也行，这里演示两次更直观
            t_sum = torch.tensor(self.loss_dict[loss].sum, dtype=torch.float32, device=device)
            t_cnt = torch.tensor(self.loss_dict[loss].count, dtype=torch.float32, device=device)

            dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)

            global_sum = t_sum.item()
            global_cnt = t_cnt.item()

            # 防止除零（极端情况：全局没有任何样本）
            if global_cnt > 0:
                self.loss_dict[loss].sum = global_sum
                self.loss_dict[loss].count = global_cnt
                self.loss_dict[loss].avg = global_sum / global_cnt
            else:
                # 保持为 0，或设为 NaN/Inf 按需处理
                self.loss_dict[loss].sum = 0.0
                self.loss_dict[loss].count = 0
                self.loss_dict[loss].avg = 0.0
    
    def __str__(self):
        return self.format_metrics()
    
    def __repr__(self):
        return self.loss_dict[self.loss_list[0]].avg
