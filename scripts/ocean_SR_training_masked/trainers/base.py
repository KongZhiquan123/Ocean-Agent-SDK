"""
Base trainer for ocean SR training (masked version).

@author Leizheng
@contributors kongzhiquan
@date 2026-02-06
@version 5.2.0

@changelog
  - 2026-02-25 Leizheng: v5.2.0 修复 __event__ 双重输出 Bug
    - _log_json_event 恢复 print 到 stdout（tqdm 默认走 stderr，无冲突）
    - 移除 self.main_log(json_str)：经 StreamHandler 写 stderr，
      TypeScript 同时解析 stdout/stderr 两个 buffer 导致事件被处理两次
    - except 块补充 epoch_bar.close() 防止异常路径资源泄漏
  - 2026-02-11 Leizheng: v5.1.1 predict() 重复推理注释说明
    - _save_test_samples() 调用处添加注释，说明与 predict 循环的重复推理问题
  - 2026-02-11 Leizheng: v5.1.0 predict() 结构化事件输出 + 错误捕获
    - predict_start / predict_progress / predict_end 事件供 TypeScript 进程管理器感知
    - try-catch 包裹并输出 training_error + phase="predict"
    - predict() 完成后调用 _save_test_samples() 生成可视化数据
  - 2026-02-11 Leizheng: v5.0.0 Patch 模式全图重建 + 独立推理流程
    - 新增 _full_coverage_positions() 全覆盖 patch 网格函数（含边界补丁）
    - 新增 _reconstruct_full_image() 方法：patch 拼接 + 重叠平均 + 反归一化
    - 改造 _save_test_samples(): patch 模式下调用全图重建，输出物理值全图
    - 新增 predict() 方法：对测试集全样本执行全图 SR 推理并保存 NPY
  - 2026-02-09 kongzhiquan: v4.5.1 test_samples.npz 追加经纬度/变量名/文件名元数据
  - 2026-02-09 kongzhiquan: v4.5.0 测试样本保存用于可视化
    - 新增 _save_test_samples() 方法，保存前 N 条测试样本的 LR/SR/HR 到 npz
    - process() 在加载最佳模型后自动调用，输出 test_samples.npz
  - 2026-02-08 Leizheng: v4.4.0 PGN+Patch 兼容 + dir() 修复
    - evaluate() 中 patch 模式下跳过 PGN decode（避免空间维度不匹配）
    - __init__ 新增 self.patch_mode 标志
    - 修复 'epoch' in dir() → 'epoch' in locals()
  - 2026-02-08 Leizheng: v4.3.0 验证 OOM 防护
    - evaluate() 前调用 torch.cuda.empty_cache() 释放碎片化缓存
    - evaluate() 改为逐 batch 计算 metrics，不再累积 all_y/all_y_pred
    - valid_loader/test_loader 在 DDP 模式下加 DistributedSampler
  - 2026-02-07 Leizheng: v4.2.0 修复事件输出通道
    - _log_json_event 直接 print 到 stdout（不再依赖 logging.info → stderr）
    - training_error 事件在所有 rank 输出（不限主进程），确保多卡崩溃可被捕获
  - 2026-02-07 Leizheng: v4.1.0 process() 添加 try-catch 结构化错误输出
    - 训练崩溃时输出 training_error 事件，包含错误类型/消息/traceback/epoch
    - 新增 _current_epoch 跟踪当前训练轮次
  - 2026-02-07 Leizheng: v4.0.0 通用模型尺寸适配
    - 新增 _pad_to_divisible() / _crop_to_original() 工具方法
    - 从 data_args 读取 model_divisor，inference() 自动 pad/crop
    - 覆盖 UNet2d 等标准模型，扩散模型由各自 diffusion.py 处理
  - 2026-02-07 Leizheng: v3.0.0 AMP 混合精度 + Gradient Checkpointing
    - 新增 use_amp / gradient_checkpointing 配置项
    - train() 使用 torch.amp.autocast + GradScaler
    - evaluate() 使用 autocast 加速推理
    - gradient checkpointing 包装 model forward 降低激活显存
    - save_ckpt / load_ckpt 保存/恢复 scaler 状态
  - 2026-02-07 kongzhiquan: v2.1.0 添加结构化日志输出
    - 训练开始/结束时输出 training_start/training_end 事件
    - 每个 epoch 输出 epoch_train/epoch_valid 事件
    - 最终评估输出 final_valid/final_test 事件
    - 所有事件使用 JSON 格式，便于报告生成脚本解析
  - 2026-02-06 Leizheng: v2.0.0 集成陆地掩码支持
    - build_data() 加载 mask_hr / mask_lr
    - build_loss() 改用 MaskedLpLoss
    - build_evaluator() 改用 MaskedEvaluator
    - train() / evaluate() 传入 mask
  - 原始版本: v1.0.0
"""

import os
import json
import torch
import wandb
import logging
import numpy as np
import torch.nn.functional as F
from datetime import datetime

import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import sys
from utils.loss import LossRecord, LpLoss, MaskedLpLoss
from utils.ddp import debug_barrier
from utils.metrics import Evaluator, MaskedEvaluator
from functools import partial
import torch.utils.checkpoint
from models import _model_dict
from datasets import _dataset_dict


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.model_args = args['model']
        self.data_args = args['data']
        self.optim_args = args['optimize']
        self.scheduler_args = args['schedule']
        self.train_args = args['train']
        self.log_args = args['log']
        
        self.set_distribute()

        # AMP 混合精度 + Gradient Checkpointing
        self.use_amp = self.train_args.get('use_amp', False) and torch.cuda.is_available()
        self.gradient_checkpointing = self.train_args.get('gradient_checkpointing', False)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # 模型整除要求（用于 inference pad/crop）
        self.model_divisor = self.data_args.get('model_divisor', 1)

        # Patch 模式标志（影响验证时是否做 decode）
        self.patch_mode = self.data_args.get('patch_size', None) is not None

        self.logger = logging.info if self.log_args.get('log', True) else print
        self.wandb = self.log_args.get('wandb', False)
        if self.check_main_process() and self.wandb:
            wandb.init(
                project=self.log_args.get('wandb_project', 'default'), 
                name=self.train_args.get('saving_name', 'experiment'),
                tags=[self.model_args.get('name', 'model'), self.data_args.get('name', 'dataset')],
                config=args)
        
        self.model_name = self.model_args['name']
        self.main_log("Building {} model".format(self.model_name))
        self.model = self.build_model()
        self.apply_init()
        
        self.start_epoch = 0
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        

        if self.train_args.get('load_ckpt', False):
            self.load_ckpt(self.train_args['ckpt_path'])
        


        self.model = self.model.to(self.device)
        
        if self.dist:
            if self.dist_mode == 'DP':
                self.device_ids = self.train_args.get('device_ids', range(torch.cuda.device_count()))
                self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
                self.main_log("Using DataParallel with GPU: {}".format(self.device_ids))
            elif self.dist_mode == 'DDP':
                self.local_rank = self.train_args.get('local_rank', 0)
                torch.cuda.set_device(self.local_rank)
                self.model = self.model.to(self.local_rank)
                self.model = DDP(
                    self.model, 
                    device_ids=[self.local_rank], 
                    output_device=self.local_rank)
        
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()
        
        self.loss_fn = self.build_loss()
        self.evaluator = self.build_evaluator()

        self.main_log("Model: {}".format(self.model))
        self.main_log("Model parameters: {:.2f}M".format(sum(p.numel() for p in self.model.parameters()) / 1e6))
        self.main_log("Optimizer: {}".format(self.optimizer))
        self.main_log("Scheduler: {}".format(self.scheduler))
        if self.use_amp:
            self.main_log("AMP mixed precision: ENABLED")
        if self.gradient_checkpointing:
            self.main_log("Gradient checkpointing: ENABLED")

        self.data = self.data_args['name']
        self.main_log("Loading {} dataset".format(self.data))
        self.build_data()
        self.main_log("Train dataset size: {}".format(len(self.train_loader.dataset)))
        self.main_log("Valid dataset size: {}".format(len(self.valid_loader.dataset)))
        self.main_log("Test dataset size: {}".format(len(self.test_loader.dataset)))

        self.epochs = self.train_args['epochs']
        self.eval_freq = self.train_args['eval_freq']
        self.patience = self.train_args['patience']
        
        self.saving_best = self.train_args.get('saving_best', True)
        self.saving_ckpt = self.train_args.get('saving_ckpt', False)
        self.ckpt_freq = self.train_args.get('ckpt_freq', 100)
        self.ckpt_max = self.train_args.get('ckpt_max', 5)
        self.saving_path = self.train_args.get('saving_path', None)

    def _unwrap(self):
        if isinstance(self.model, (DDP, nn.DataParallel)):
            return self.model.module
        return self.model
    
    def set_distribute(self):
        self.dist = self.train_args.get('distribute', False)
        if self.dist:
            self.dist_mode = self.train_args.get('distribute_mode', 'DDP')
        if self.dist and self.dist_mode == 'DDP':
            self.local_rank = self.train_args.get('local_rank', 0)
            self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    def get_initializer(self, name):
        if name is None:
            return None
        
        if name == 'xavier_normal':
            init_ = partial(torch.nn.init.xavier_normal_)
        elif name == 'kaiming_uniform':
            init_ = partial(torch.nn.init.kaiming_uniform_)
        elif name == 'kaiming_normal':
            init_ = partial(torch.nn.init.kaiming_normal_)
        return init_

    def apply_init(self, **kwargs):
        initializer = self.get_initializer(self.train_args.get('initializer', None))
        if initializer is not None:
            self.model.apply(initializer)
            self.main_log("Apply {} initializer".format(self.train_args.get('initializer', None)))
    
    def build_optimizer(self, **kwargs):
        if self.optim_args['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                weight_decay=self.optim_args['weight_decay'],
            )
        elif self.optim_args['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                momentum=self.optim_args['momentum'],
                weight_decay=self.optim_args['weight_decay'],
            )
        elif self.optim_args['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                weight_decay=self.optim_args['weight_decay'],
            )
        else:
            raise NotImplementedError("Optimizer {} not implemented".format(self.optim_args['optimizer']))
        return optimizer
    
    def build_scheduler(self, **kwargs):
        if self.scheduler_args['scheduler'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_args['milestones'],
                gamma=self.scheduler_args['gamma'],
            )
        elif self.scheduler_args['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optim_args['lr'],
                div_factor=self.scheduler_args['div_factor'],
                final_div_factor=self.scheduler_args['final_div_factor'],
                pct_start=self.scheduler_args['pct_start'],
                steps_per_epoch=self.scheduler_args['steps_per_epoch'],
                epochs=self.train_args['epochs'],
            )
        elif self.scheduler_args['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_args['step_size'],
                gamma=self.scheduler_args['gamma'],
            )
        else:
            scheduler = None
            if self.scheduler_args['scheduler'] is not None:
                raise NotImplementedError("Scheduler {} not implemented".format(self.scheduler_args['scheduler']))
            
        return scheduler
    
    def build_model(self, **kwargs):
        if self.model_name not in _model_dict:
            raise NotImplementedError("Model {} not implemented".format(self.model_name))
        model = _model_dict[self.model_name](self.model_args)
        return model
    
    def build_loss(self, **kwargs):
        loss_fn = MaskedLpLoss(size_average=False)
        return loss_fn

    def build_evaluator(self):
        return MaskedEvaluator(shape=self.data_args['shape'])
    
    def build_data(self, **kwargs):
        if self.data_args['name'] not in _dataset_dict:
            raise NotImplementedError("Dataset {} not implemented".format(self.data_args['name']))
        dataset = _dataset_dict[self.data_args['name']](self.data_args)
        if self.dist and self.dist_mode == 'DDP':
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset.train_dataset,
                shuffle=True,
                drop_last=True,
                )
            # 验证/测试集也使用 DistributedSampler，避免每个 rank 重复处理全部数据
            self.valid_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset.valid_dataset,
                shuffle=False,
                drop_last=False,
                )
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset.test_dataset,
                shuffle=False,
                drop_last=False,
                )
            shuffle = False
        else:
            self.train_sampler = None
            self.valid_sampler = None
            self.test_sampler = None
            shuffle = True

        self.train_loader = torch.utils.data.DataLoader(
            dataset.train_dataset,
            batch_size=self.data_args.get('train_batchsize', 10),
            shuffle=shuffle,
            num_workers=self.data_args.get('num_workers', 0),
            sampler=self.train_sampler,
            drop_last=True,
            pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(
            dataset.valid_dataset,
            batch_size=self.data_args.get('eval_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
            sampler=self.valid_sampler,
            pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset.test_dataset,
            batch_size=self.data_args.get('eval_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
            sampler=self.test_sampler,
            pin_memory=True)
        
        self.normalizer = dataset.normalizer

        # 加载陆地掩码（如果数据集提供了 mask）
        if hasattr(dataset, 'mask_hr') and dataset.mask_hr is not None:
            self.mask_hr = dataset.mask_hr.to(self.device)
            self.main_log("Loaded HR mask: {} ocean pixels / {} total".format(
                int(self.mask_hr.sum().item()), self.mask_hr.numel()))
        else:
            self.mask_hr = None

        if hasattr(dataset, 'mask_lr') and dataset.mask_lr is not None:
            self.mask_lr = dataset.mask_lr.to(self.device)
        else:
            self.mask_lr = None
    
    def _get_state_dict_cpu(self):
        if self.dist and self.dist_mode == 'DDP':
            model_to_save = self.model.module
        elif isinstance(self.model, torch.nn.DataParallel):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        return {k: v.detach().cpu() for k, v in model_to_save.state_dict().items()}
    
    def save_ckpt(self, epoch):
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        state_dict_cpu = self._get_state_dict_cpu()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict_cpu,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
        }, os.path.join(self.saving_path, f"model_epoch_{epoch}.pth"))
        if self.ckpt_max is not None and self.ckpt_max > 0:
            ckpt_list = [f for f in os.listdir(self.saving_path) if f.startswith('model_epoch_') and f.endswith('.pth')]
            if len(ckpt_list) > self.ckpt_max:
                ckpt_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                os.remove(os.path.join(self.saving_path, ckpt_list[0]))
                    
    def save_model(self, model_path):
        state_dict_cpu = self._get_state_dict_cpu()
        torch.save(state_dict_cpu, model_path)
        self.main_log("Save model to {}".format(model_path))
        
    def load_model(self, model_path):
        state = torch.load(model_path, map_location="cpu")
        if self.dist and self.dist_mode == 'DDP':
            self.model.module.load_state_dict(state)
        elif isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state)
        else:
            self.model.load_state_dict(state)
        self.main_log("Load model from {}".format(model_path))
    
    def load_ckpt(self, ckpt_path):        
        state = torch.load(ckpt_path, map_location="cpu")
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            # ✅ 强制把optimizer中的状态迁移到GPU
            for state_tensor in self.optimizer.state.values():
                for k, v in state_tensor.items():
                    if isinstance(v, torch.Tensor):
                        state_tensor[k] = v.to(self.device)
        if 'scheduler_state_dict' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        if 'scaler_state_dict' in state and state['scaler_state_dict'] is not None and self.use_amp:
            self.scaler.load_state_dict(state['scaler_state_dict'])
        self.start_epoch = state.get('epoch', 0) + 1
        self.main_log("Load checkpoint from {}, epoch {}".format(ckpt_path, state.get('epoch', 'N/A')))
    
    def check_main_process(self):
        if self.dist is False:
            return True
        if self.dist_mode == 'DP':
            return True
        if self.local_rank == 0:
            return True
        return False
    
    def main_log(self, msg):
        if self.check_main_process():
            self.logger(msg)
    
    def _log_json_event(self, event_type: str, **data):
        """输出结构化 JSON 日志事件

        事件通过 stdout 发出 __event__JSON__event__ 标记，
        由 TypeScript 进程管理器解析。
        tqdm 默认渲染到 stderr，stdout 与其不冲突，无需重定向。

        - training_error: 任何 rank 都输出（崩溃可能在非主进程）
        - 其他事件: 仅主进程输出（避免多卡重复）
        """
        event = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        json_str = f"__event__{json.dumps(event, ensure_ascii=False)}__event__"
        if event_type == "training_error":
            # 错误事件：所有 rank 都输出（崩溃可能发生在任意 rank）
            print(json_str, flush=True)
        elif self.check_main_process():
            # 普通事件：仅主进程输出
            print(json_str, flush=True)

    def process(self, **kwargs):
        training_start_time = datetime.now()
        self.main_log("Start training")
        self._current_epoch = None
        self._batch_bar = None
        _is_main = self.check_main_process()
        _ew = len(str(self.epochs))  # epoch 数字宽度，用于对齐

        try:
            # 训练开始事件
            self._log_json_event(
                "training_start",
                model_name=self.model_name,
                model_params=sum(p.numel() for p in self.model.parameters()) / 1e6,
                dataset_name=self.data,
                train_samples=len(self.train_loader.dataset),
                valid_samples=len(self.valid_loader.dataset),
                test_samples=len(self.test_loader.dataset),
                total_epochs=self.epochs,
                batch_size=self.data_args.get('train_batchsize', 10),
                learning_rate=self.optim_args['lr'],
                optimizer=self.optim_args['optimizer'],
                patience=self.patience,
                eval_freq=self.eval_freq,
                device=str(self.device),
                distribute=self.dist,
                distribute_mode=getattr(self, 'dist_mode', None),
                mask_hr_info={
                    "ocean_pixels": int(self.mask_hr.sum().item()) if self.mask_hr is not None else None,
                    "total_pixels": self.mask_hr.numel() if self.mask_hr is not None else None,
                } if self.mask_hr is not None else None,
            )

            best_epoch = 0
            best_metrics = None
            best_path = os.path.join(self.saving_path, "best_model.pth")
            counter = 0
            early_stopped = False
            epoch_history = []

            if dist.is_initialized():
                dist.barrier()

            # 外层 epoch 进度条（贯穿整个训练，leave=True 保留在终端）
            epoch_bar = tqdm(
                total=self.epochs - self.start_epoch,
                desc="Epochs",
                unit="ep",
                leave=True,
                dynamic_ncols=True,
                disable=not _is_main,
            )

            # logging_redirect_tqdm 使 logging.info() 通过 tqdm.write() 输出，避免破坏进度条
            with logging_redirect_tqdm():
                if _is_main:
                    n_train = len(self.train_loader.dataset)
                    n_valid = len(self.valid_loader.dataset)
                    n_test  = len(self.test_loader.dataset)
                    tqdm.write(
                        f"\n{'═' * 56}\n"
                        f"  {self.model_name}"
                        f"  ·  train {n_train}  ·  val {n_valid}  ·  test {n_test}\n"
                        f"{'═' * 56}"
                    )

                for epoch in range(self.start_epoch, self.epochs):
                    self._current_epoch = epoch
                    ep_label = f"[{epoch + 1:>{_ew}d}/{self.epochs}]"

                    # 内层 batch 进度条（每个 epoch 结束后自动消失，leave=False）
                    _orig_loader = self.train_loader
                    if _is_main:
                        self._batch_bar = tqdm(
                            _orig_loader,
                            desc=f"{ep_label} train",
                            leave=False,
                            dynamic_ncols=True,
                        )
                        self.train_loader = self._batch_bar
                    else:
                        self._batch_bar = None

                    try:
                        train_loss_record = self.train(epoch)
                    finally:
                        self.train_loader = _orig_loader
                        if self._batch_bar is not None:
                            self._batch_bar.close()
                        self._batch_bar = None

                    lr = self.optimizer.param_groups[0]["lr"]
                    train_dict = train_loss_record.to_dict()
                    elapsed = train_loss_record.elapsed()

                    # epoch 训练摘要（tqdm.write 不会破坏进度条位置）
                    if _is_main:
                        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in train_dict.items())
                        tqdm.write(f"{ep_label}  train  {metrics_str}  lr={lr:.2e}  t={elapsed:.1f}s")

                    # 更新外层 epoch bar postfix
                    train_loss_val = next(iter(train_dict.values())) if train_dict else 0.0
                    epoch_bar.set_postfix({"loss": f"{train_loss_val:.4f}", "lr": f"{lr:.2e}"}, refresh=False)
                    epoch_bar.update(1)

                    # 结构化事件
                    self._log_json_event("epoch_train", epoch=epoch, metrics=train_dict, lr=lr)

                    if _is_main and self.wandb:
                        wandb.log(train_dict)

                    if _is_main and self.saving_ckpt and (epoch + 1) % self.ckpt_freq == 0:
                        self.save_ckpt(epoch)
                        tqdm.write(f"{ep_label}  ckpt saved → {self.saving_path}")

                    if (epoch + 1) % self.eval_freq == 0:
                        valid_loss_record = self.evaluate(split="valid")
                        valid_metrics = valid_loss_record.to_dict()

                        is_best = not best_metrics or valid_metrics['valid_loss'] < best_metrics['valid_loss']
                        self._log_json_event("epoch_valid", epoch=epoch, metrics=valid_metrics, is_best=is_best)

                        epoch_history.append({
                            "epoch": epoch,
                            "train_loss": train_dict.get('train_loss'),
                            "valid_metrics": valid_metrics,
                        })

                        if _is_main:
                            val_str = "  ".join(f"{k}={v:.4f}" for k, v in valid_metrics.items())
                            best_mark = "  ★ best" if is_best else ""
                            tqdm.write(f"{ep_label}  valid  {val_str}{best_mark}")

                        if _is_main and self.wandb:
                            wandb.log(valid_metrics)

                        if is_best:
                            counter = 0
                            best_epoch = epoch
                            best_metrics = valid_metrics
                            # 更新 epoch bar 显示 val loss
                            val_loss_val = next(iter(valid_metrics.values())) if valid_metrics else 0.0
                            epoch_bar.set_postfix(
                                {"loss": f"{train_loss_val:.4f}", "val": f"{val_loss_val:.4f}", "best": f"{epoch + 1}"},
                                refresh=False,
                            )
                            if _is_main and self.saving_best:
                                self.save_model(best_path)
                        elif self.patience != -1:
                            counter += 1
                            if counter >= self.patience:
                                early_stopped = True
                                tqdm.write(f"{ep_label}  early stop (patience={self.patience})")
                                self._log_json_event("early_stop", epoch=epoch, patience=self.patience)
                                if not self.dist:
                                    break
                                stop_flag = torch.tensor(0, device=self.device)
                                if _is_main and self.patience != -1 and counter >= self.patience:
                                    stop_flag += 1
                                if self.dist and dist.is_initialized():
                                    dist.broadcast(stop_flag, src=0)
                                if stop_flag.item() > 0:
                                    break

            epoch_bar.close()

            if _is_main:
                total_sec = (datetime.now() - training_start_time).total_seconds()
                h, rem = divmod(int(total_sec), 3600)
                m, s = divmod(rem, 60)
                tqdm.write(
                    f"\n{'═' * 56}\n"
                    f"  完成  best_epoch={best_epoch + 1}  总耗时={h}h {m}m {s}s\n"
                    f"{'═' * 56}\n"
                )

            self.main_log("Optimization Finished!")

            if _is_main and not best_metrics:
                self.save_model(best_path)

            if self.dist and dist.is_initialized():
                dist.barrier()

            self.load_model(best_path)

            # 保存测试集前几条样本的 LR/SR/HR 用于可视化对比
            self._save_test_samples(num_samples=2)

            valid_loss_record = self.evaluate(split="valid")
            self.main_log("Valid metrics: {}".format(valid_loss_record))
            self._log_json_event("final_valid", metrics=valid_loss_record.to_dict(), best_epoch=best_epoch)

            test_loss_record = self.evaluate(split="test")
            self.main_log("Test metrics: {}".format(test_loss_record))
            self._log_json_event("final_test", metrics=test_loss_record.to_dict(), best_epoch=best_epoch)

            # 训练结束事件
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            actual_epochs = epoch + 1 if 'epoch' in locals() else self.epochs
            self._log_json_event(
                "training_end",
                training_duration_seconds=training_duration,
                actual_epochs=actual_epochs,
                best_epoch=best_epoch,
                early_stopped=early_stopped,
                final_valid_metrics=valid_loss_record.to_dict(),
                final_test_metrics=test_loss_record.to_dict(),
            )

            if _is_main and self.wandb:
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary.update(test_loss_record.to_dict())
                wandb.finish()

            if self.dist and dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if 'epoch_bar' in locals():
                epoch_bar.close()
            import traceback
            tb = traceback.format_exc()
            error_type = type(e).__name__
            self.main_log(f"[FATAL] Training crashed: {error_type}: {e}")
            self.main_log(tb)
            self._log_json_event(
                "training_error",
                error_type=error_type,
                error_message=str(e),
                traceback=tb,
                epoch=getattr(self, '_current_epoch', None),
            )
            raise

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
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if self.gradient_checkpointing:
                    y_pred = torch.utils.checkpoint.checkpoint(
                        self.model, x, use_reentrant=False
                    )
                else:
                    y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y, mask=mask_hr)
            loss_record.update({"train_loss": loss.sum().item()}, n=x.size(0))
            loss = loss.mean()
            # 实时更新内层 batch 进度条的 loss 显示
            if self._batch_bar is not None:
                self._batch_bar.set_postfix(
                    {"loss": f"{loss_record.to_dict()['train_loss']:.4f}"}, refresh=False
                )
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss_record
    
    def _pad_to_divisible(self, x, divisor, channel_last=True):
        """Pad 张量到能被 divisor 整除的尺寸（reflect 模式）。

        Args:
            x: 输入张量
            divisor: 整除因子
            channel_last: True=[B,H,W,C], False=[B,C,H,W]

        Returns:
            (padded_x, orig_h, orig_w)
        """
        if divisor <= 1:
            if channel_last:
                return x, x.shape[1], x.shape[2]
            else:
                return x, x.shape[2], x.shape[3]
        if channel_last:  # [B, H, W, C]
            h, w = x.shape[1], x.shape[2]
            pad_h = (divisor - h % divisor) % divisor
            pad_w = (divisor - w % divisor) % divisor
            if pad_h or pad_w:
                x = x.permute(0, 3, 1, 2)  # -> [B,C,H,W]
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
                x = x.permute(0, 2, 3, 1)  # -> [B,H+pad,W+pad,C]
        else:  # [B, C, H, W]
            h, w = x.shape[2], x.shape[3]
            pad_h = (divisor - h % divisor) % divisor
            pad_w = (divisor - w % divisor) % divisor
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, h, w

    def _crop_to_original(self, x, h, w, channel_last=True):
        """Crop 张量回原始尺寸。

        Args:
            x: padded 张量
            h: 原始高度
            w: 原始宽度
            channel_last: True=[B,H,W,C], False=[B,C,H,W]
        """
        if channel_last:  # [B, H, W, C]
            return x[:, :h, :w, :]
        else:  # [B, C, H, W]
            return x[:, :, :h, :w]

    def inference(self, x, y, **kwargs):
        x, orig_h, orig_w = self._pad_to_divisible(x, self.model_divisor, channel_last=True)
        result = self.model(x)
        result = self._crop_to_original(result, y.shape[1], y.shape[2], channel_last=True)
        return result.reshape(y.shape)
    
    def evaluate(self, split="valid", **kwargs):
        if split == "valid":
            eval_loader = self.valid_loader
        elif split == "test":
            eval_loader = self.test_loader
        else:
            raise ValueError("split must be 'valid' or 'test'")

        # 释放训练阶段留下的碎片化 CUDA 缓存
        # 验证（尤其是扩散模型采样）需要大块连续显存，碎片化会导致 OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        loss_record = self.evaluator.init_record(["{}_loss".format(split)])
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            for batch in eval_loader:
                # patch 模式返回 (x, y, mask_patch)，全图模式返回 (x, y)
                if len(batch) == 3:
                    x, y, mask_patch = batch
                    mask_patch = mask_patch.to(self.device, non_blocking=True)
                else:
                    x, y = batch
                    mask_patch = self.mask_hr  # 全局 mask（全图模式）
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y_pred = self.inference(x, y, **kwargs)
                # normalizer 可能是 dict {'hr': ..., 'lr': ...} 或单个对象
                _norm = self.normalizer['hr'] if isinstance(self.normalizer, dict) else self.normalizer
                # Patch 模式下 PGN 的 mean/std 形状是全图空间维度，与 patch 维度不匹配
                # 此时在归一化空间中计算 metrics（仍然有效，因为值域一致）
                # 全图模式下正常 decode 回原始数据空间
                if not self.patch_mode and _norm is not None:
                    y_pred = _norm.decode(y_pred)
                    y = _norm.decode(y)
                # 逐 batch 计算 loss 和指标，避免在 GPU 上累积全部验证结果
                batch_loss = self.loss_fn(y_pred, y, mask=mask_patch)
                loss_record.update({"{}_loss".format(split): batch_loss.item()})
                self.evaluator(y_pred, y, record=loss_record, mask=mask_patch)
                del y_pred, batch_loss  # 立即释放显存
        if self.dist and dist.is_initialized():
            loss_record.dist_reduce()
        return loss_record

    @staticmethod
    def _full_coverage_positions(H, W, ps):
        """生成确保全覆盖的 patch 位置列表（含边界补丁）。

        使用非重叠步进，若尾部不整除则额外添加从末尾对齐的补丁，
        重叠区域在拼接时取平均。

        Args:
            H: 全图高度
            W: 全图宽度
            ps: patch 尺寸

        Returns:
            list of (top, left) 元组
        """
        def _axis_positions(length, size):
            pos = list(range(0, length - size + 1, size))
            if not pos or pos[-1] + size < length:
                pos.append(max(0, length - size))
            return sorted(set(pos))

        rows = _axis_positions(H, ps)
        cols = _axis_positions(W, ps)
        return [(r, c) for r in rows for c in cols]

    def _reconstruct_full_image(self, sample_idx, split='test'):
        """对单个时间步做全图 SR 重建（patch 拼接 + 反归一化）。

        绕过 DataLoader 的 patch 裁剪，直接从 dataset 取全图数据，
        切成全覆盖 patch → 逐 patch 推理 → 重叠平均拼接 → decode 反归一化。

        Args:
            sample_idx: 时间步索引（在原始样本序列中的位置）
            split: 数据集 split（默认 'test'）

        Returns:
            dict with keys 'lr', 'sr', 'hr' — numpy arrays in physical value space
                lr: [h, w, C], sr: [H, W, C], hr: [H, W, C]
        """
        dataset = self.test_loader.dataset

        # 1. 拿到全图数据（归一化后的张量）
        x_full = dataset.x[sample_idx]   # [h, w, C]
        y_full = dataset.y[sample_idx]   # [H, W, C]

        H, W, C = y_full.shape
        ps = dataset.patch_size
        scale = dataset.scale

        # 2. 计算全覆盖 grid
        positions = self._full_coverage_positions(H, W, ps)

        # 3. 逐 patch 推理，累加到全图画布
        canvas_sum = torch.zeros(H, W, C)
        canvas_cnt = torch.zeros(H, W, 1)

        for (top, left) in positions:
            # 裁剪 HR patch
            y_patch = y_full[top:top+ps, left:left+ps, :]

            # 推导对应的 LR patch 坐标
            lr_ps = ps // scale
            lr_top = top // scale
            lr_left = left // scale
            x_patch = x_full[lr_top:lr_top+lr_ps, lr_left:lr_left+lr_ps, :]

            # 推理（添加 batch 维度）
            x_batch = x_patch.unsqueeze(0).to(self.device)
            y_batch = y_patch.unsqueeze(0).to(self.device)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                y_pred = self.inference(x_batch, y_batch)

            # 累加到画布
            canvas_sum[top:top+ps, left:left+ps, :] += y_pred[0].cpu().float()
            canvas_cnt[top:top+ps, left:left+ps, :] += 1

        # 4. 平均重叠区域
        sr_full = canvas_sum / canvas_cnt.clamp(min=1)

        # 5. decode 反归一化（全图尺寸，PGN 维度匹配）
        norm_hr = self.normalizer.get('hr') if isinstance(self.normalizer, dict) else self.normalizer
        norm_lr = self.normalizer.get('lr') if isinstance(self.normalizer, dict) else self.normalizer

        if norm_hr is not None:
            sr_dec = norm_hr.decode(sr_full.unsqueeze(0)).squeeze(0)  # [H, W, C]
            hr_dec = norm_hr.decode(y_full.unsqueeze(0)).squeeze(0)
        else:
            sr_dec = sr_full
            hr_dec = y_full

        if norm_lr is not None:
            lr_dec = norm_lr.decode(x_full.unsqueeze(0)).squeeze(0)   # [h, w, C]
        else:
            lr_dec = x_full

        return {
            'lr': lr_dec.cpu().float().numpy(),   # [h, w, C] 物理值
            'sr': sr_dec.cpu().float().numpy(),   # [H, W, C] 物理值
            'hr': hr_dec.cpu().float().numpy(),   # [H, W, C] 物理值
        }

    def _save_test_samples(self, num_samples=2):
        """保存测试集前 num_samples 条样本的 LR / SR / HR 数据用于可视化。

        patch_mode=True 时使用全图重建（patch 拼接 + 反归一化），
        输出为物理值全图而非裁剪的归一化 patch。

        输出文件: {saving_path}/test_samples.npz
        包含:
            lr  - 低分辨率输入 [N, H_lr, W_lr, C]  物理值
            sr  - 模型推理输出 [N, H_hr, W_hr, C]  物理值
            hr  - 高分辨率真值 [N, H_hr, W_hr, C]  物理值
            mask_hr - (可选) 陆地掩码 [1, H_hr, W_hr, 1]
        """
        if not self.check_main_process():
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.eval()
        saved_lr, saved_sr, saved_hr = [], [], []

        dataset = self.test_loader.dataset
        # 真实样本数（patch 模式下 len(dataset) 是 patch 总数）
        if self.patch_mode and hasattr(dataset, '_grid_positions') and dataset._grid_positions is not None:
            n_real_samples = len(dataset.x)
        else:
            n_real_samples = len(dataset)
        actual_num = min(num_samples, n_real_samples)

        with torch.no_grad():
            if self.patch_mode:
                # 全图重建模式：逐样本 patch 拼接 + 反归一化
                for i in range(actual_num):
                    result = self._reconstruct_full_image(i)
                    saved_lr.append(result['lr'])
                    saved_sr.append(result['sr'])
                    saved_hr.append(result['hr'])
            else:
                # 全图训练模式：直接 decode
                count = 0
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    for batch in self.test_loader:
                        if len(batch) == 3:
                            x, y, _ = batch
                        else:
                            x, y = batch

                        x = x.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)
                        y_pred = self.inference(x, y)

                        _norm_hr = self.normalizer.get('hr') if isinstance(self.normalizer, dict) else self.normalizer
                        _norm_lr = self.normalizer.get('lr') if isinstance(self.normalizer, dict) else self.normalizer

                        if _norm_hr is not None:
                            y_pred_dec = _norm_hr.decode(y_pred)
                            y_dec = _norm_hr.decode(y)
                        else:
                            y_pred_dec = y_pred
                            y_dec = y

                        if _norm_lr is not None:
                            x_dec = _norm_lr.decode(x)
                        else:
                            x_dec = x

                        bs = x.size(0)
                        for i in range(bs):
                            if count >= actual_num:
                                break
                            saved_lr.append(x_dec[i].detach().cpu().float().numpy())
                            saved_sr.append(y_pred_dec[i].detach().cpu().float().numpy())
                            saved_hr.append(y_dec[i].detach().cpu().float().numpy())
                            count += 1
                        if count >= actual_num:
                            break

        save_data = {
            'lr': np.array(saved_lr),
            'sr': np.array(saved_sr),
            'hr': np.array(saved_hr),
        }

        # 保存全图 mask（不再保存 per-patch mask）
        if self.mask_hr is not None:
            save_data['mask_hr'] = self.mask_hr.cpu().numpy()  # [1, H, W, 1]

        # 保存元数据（经纬度、文件名、变量名）— 使用全图坐标
        test_ds = self.test_loader.dataset
        if hasattr(test_ds, 'get_meta'):
            # patch_idx=None 使其返回全图坐标：直接用 sample_idx=0
            # 对于 patch 模式，传 idx=0 并手动获取全图元数据
            if self.patch_mode:
                # 直接访问 dataset 的全图属性，绕过 patch 裁剪
                if test_ds.lon_hr is not None:
                    save_data['lon_hr'] = test_ds.lon_hr
                if test_ds.lat_hr is not None:
                    save_data['lat_hr'] = test_ds.lat_hr
                if test_ds.lon_lr is not None:
                    save_data['lon_lr'] = test_ds.lon_lr
                if test_ds.lat_lr is not None:
                    save_data['lat_lr'] = test_ds.lat_lr
                if test_ds.dyn_vars is not None:
                    save_data['dyn_vars'] = np.array(test_ds.dyn_vars)
                if test_ds.filenames is not None and len(test_ds.filenames) > 0:
                    save_data['filename'] = np.array(test_ds.filenames[0])
            else:
                meta = test_ds.get_meta(0)
                if meta['lon_hr'] is not None:
                    save_data['lon_hr'] = meta['lon_hr']
                if meta['lat_hr'] is not None:
                    save_data['lat_hr'] = meta['lat_hr']
                if meta['lon_lr'] is not None:
                    save_data['lon_lr'] = meta['lon_lr']
                if meta['lat_lr'] is not None:
                    save_data['lat_lr'] = meta['lat_lr']
                if meta['dyn_vars'] is not None:
                    save_data['dyn_vars'] = np.array(meta['dyn_vars'])
                if meta['filename'] is not None:
                    save_data['filename'] = np.array(meta['filename'])

        save_path = os.path.join(self.saving_path, 'test_samples.npz')
        np.savez(save_path, **save_data)
        self.main_log("Saved {} test samples (LR/SR/HR) to {}".format(actual_num, save_path))

    def predict(self, output_dir=None):
        """对测试集执行全图推理，保存 SR 输出到 NPY 文件。

        patch_mode=True 时使用全图重建（patch 拼接 + 反归一化），
        否则直接推理全图。所有输出为物理值空间。

        输出结构化事件供 TypeScript 进程管理器感知：
        - predict_start: 推理启动（TypeScript waitForEvent 等待此事件）
        - predict_progress: 逐样本进度
        - predict_end: 推理完成
        - training_error + phase="predict": 异常时

        Args:
            output_dir: 输出目录，默认 {saving_path}/predictions/
        """
        if not self.check_main_process():
            return
        output_dir = output_dir or os.path.join(self.saving_path, 'predictions')
        os.makedirs(output_dir, exist_ok=True)

        dataset = self.test_loader.dataset
        # 真实样本数（patch 模式下 len(dataset) 是 patch 总数，需要用原始样本数）
        n_samples = len(dataset.x)

        try:
            # 发射 predict_start 事件（TypeScript 等待此事件确认启动成功）
            self._log_json_event("predict_start",
                n_samples=n_samples, output_dir=output_dir,
                patch_mode=self.patch_mode, model_name=self.model_name)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model.eval()
            saved_files = []

            with torch.no_grad():
                for i in range(n_samples):
                    if self.patch_mode:
                        result = self._reconstruct_full_image(i)
                    else:
                        # 全图模式直接推理
                        x = dataset.x[i]   # [h, w, C]
                        y = dataset.y[i]   # [H, W, C]
                        x_b = x.unsqueeze(0).to(self.device)
                        y_b = y.unsqueeze(0).to(self.device)
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            y_pred = self.inference(x_b, y_b)

                        norm_hr = self.normalizer.get('hr') if isinstance(self.normalizer, dict) else self.normalizer
                        if norm_hr is not None:
                            sr_dec = norm_hr.decode(y_pred).squeeze(0).cpu().float().numpy()
                            hr_dec = norm_hr.decode(y_b).squeeze(0).cpu().float().numpy()
                        else:
                            sr_dec = y_pred.squeeze(0).cpu().float().numpy()
                            hr_dec = y_b.squeeze(0).cpu().float().numpy()
                        result = {'sr': sr_dec, 'hr': hr_dec}

                    # 保存
                    fname = dataset.filenames[i] if hasattr(dataset, 'filenames') and dataset.filenames else f'{i:06d}'
                    np.save(os.path.join(output_dir, f'{fname}_sr.npy'), result['sr'])
                    saved_files.append(f'{fname}_sr.npy')

                    self.main_log(f"Predicted {i+1}/{n_samples}: {fname}")

                    # 进度事件
                    self._log_json_event("predict_progress",
                        current=i+1, total=n_samples, filename=fname)

            # 保存测试样本用于可视化（复用已有逻辑）
            # NOTE: _save_test_samples() 会对前 N 个样本独立执行推理，
            # 与上面 predict 循环存在重复推理。这是已知的效率问题。
            # 后续优化方向：将 predict 循环已得到的 SR 结果缓存后传给
            # _save_test_samples()，避免二次推理。当前改动需改变
            # _save_test_samples() 接口，暂保留现状。
            self._save_test_samples(num_samples=min(4, n_samples))

            self._log_json_event("predict_end",
                n_samples=n_samples, output_dir=output_dir,
                saved_files=saved_files)
            self.main_log(f"Predictions saved to {output_dir}")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_type = type(e).__name__
            self.main_log(f"[FATAL] Predict crashed: {error_type}: {e}")
            self.main_log(tb)
            self._log_json_event(
                "training_error",
                error_type=error_type,
                error_message=str(e),
                traceback=tb,
                phase="predict",
            )
            raise
