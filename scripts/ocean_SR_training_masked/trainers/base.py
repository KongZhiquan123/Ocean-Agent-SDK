"""
Base trainer for ocean SR training (masked version).

@author Leizheng
@contributors kongzhiquan
@date 2026-02-06
@version 4.2.0

@changelog
  - 2026-02-07 kongzhiquan: v4.2.0 修复事件输出通道
    - _log_json_event 直接 print 到 stdout（不再依赖 logging.info → stderr）
    - training_error 事件在所有 rank 输出（不限主进程），确保多卡崩溃可被捕获
  - 2026-02-07 kongzhiquan: v4.1.0 process() 添加 try-catch 结构化错误输出
    - 训练崩溃时输出 training_error 事件，包含错误类型/消息/traceback/epoch
    - 新增 _current_epoch 跟踪当前训练轮次
  - 2026-02-07 kongzhiquan: v4.0.0 通用模型尺寸适配
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
import torch.nn.functional as F
from datetime import datetime

import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
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
            shuffle = False
        else:
            self.train_sampler = None
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
            pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset.test_dataset,
            batch_size=self.data_args.get('eval_batchsize', 10),
            shuffle=False,
            num_workers=self.data_args.get('num_workers', 0),
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

        - training_error: 任何 rank 都输出（崩溃可能在非主进程）
        - 其他事件: 仅主进程输出（避免多卡重复）
        """
        event = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        json_str = f"__event__{json.dumps(event, ensure_ascii=False)}__event__"
        # 直接 print 到 stdout，确保 TypeScript 能通过 stdout pipe 捕获
        if event_type == "training_error":
            # 错误事件：所有 rank 都输出（崩溃可能发生在任意 rank）
            print(json_str, flush=True)
        elif self.check_main_process():
            # 普通事件：仅主进程输出
            print(json_str, flush=True)
        # 同时记录到日志系统（仅主进程，用于 Python 侧 train.log）
        self.main_log(json_str)

    def process(self, **kwargs):
        training_start_time = datetime.now()
        self.main_log("Start training")
        self._current_epoch = None
        try:
            # 输出训练开始事件
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
            bar = tqdm(total=self.epochs - self.start_epoch) if self.check_main_process() else None

            for epoch in range(self.start_epoch, self.epochs):
                self._current_epoch = epoch
                train_loss_record = self.train(epoch)
                lr = self.optimizer.param_groups[0]["lr"]
                self.main_log("Epoch {} | {} | lr: {:.4f}".format(epoch, train_loss_record, lr))

                # 输出 epoch 训练事件
                self._log_json_event(
                    "epoch_train",
                    epoch=epoch,
                    metrics=train_loss_record.to_dict(),
                    lr=lr,
                )

                if self.check_main_process() and self.wandb:
                    wandb.log(train_loss_record.to_dict())

                if self.check_main_process() and self.saving_ckpt and (epoch + 1) % self.ckpt_freq == 0:
                    self.save_ckpt(epoch)
                    self.main_log("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))

                if (epoch + 1) % self.eval_freq == 0:
                    valid_loss_record = self.evaluate(split="valid")
                    self.main_log("Epoch {} | {}".format(epoch, valid_loss_record))
                    valid_metrics = valid_loss_record.to_dict()

                    # 输出 epoch 验证事件
                    is_best = not best_metrics or valid_metrics['valid_loss'] < best_metrics['valid_loss']
                    self._log_json_event(
                        "epoch_valid",
                        epoch=epoch,
                        metrics=valid_metrics,
                        is_best=is_best,
                    )

                    # 记录历史
                    epoch_history.append({
                        "epoch": epoch,
                        "train_loss": train_loss_record.to_dict().get('train_loss'),
                        "valid_metrics": valid_metrics,
                    })

                    if self.check_main_process() and self.wandb:
                        wandb.log(valid_loss_record.to_dict())

                    if is_best:
                        counter = 0
                        best_epoch = epoch
                        best_metrics = valid_metrics
                        if self.check_main_process() and self.saving_best:
                            self.save_model(best_path)
                    elif self.patience != -1:
                        counter += 1
                        if counter >= self.patience:
                            early_stopped = True
                            self.main_log("Early stop at epoch {}".format(epoch))
                            self._log_json_event("early_stop", epoch=epoch, patience=self.patience)
                            if not self.dist:
                                break
                            stop_flag = torch.tensor(0, device=self.device)
                            if self.check_main_process():
                                if self.patience != -1 and counter >= self.patience:
                                    stop_flag += 1
                            if self.dist and dist.is_initialized():
                                dist.broadcast(stop_flag, src=0)
                            if stop_flag.item() > 0:
                                break
                if self.check_main_process():
                    bar.update(1)
            if self.check_main_process():
                if bar is not None:
                    bar.close()
            self.main_log("Optimization Finished!")

            if self.check_main_process() and not best_metrics:
                self.save_model(best_path)

            if self.dist and dist.is_initialized():
                dist.barrier()

            self.load_model(best_path)

            valid_loss_record = self.evaluate(split="valid")
            self.main_log("Valid metrics: {}".format(valid_loss_record))
            self._log_json_event("final_valid", metrics=valid_loss_record.to_dict(), best_epoch=best_epoch)

            test_loss_record = self.evaluate(split="test")
            self.main_log("Test metrics: {}".format(test_loss_record))
            self._log_json_event("final_test", metrics=test_loss_record.to_dict(), best_epoch=best_epoch)

            # 输出训练结束事件
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            actual_epochs = epoch + 1 if 'epoch' in dir() else self.epochs
            self._log_json_event(
                "training_end",
                training_duration_seconds=training_duration,
                actual_epochs=actual_epochs,
                best_epoch=best_epoch,
                early_stopped=early_stopped,
                final_valid_metrics=valid_loss_record.to_dict(),
                final_test_metrics=test_loss_record.to_dict(),
            )

            if self.check_main_process() and self.wandb:
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary.update(test_loss_record.to_dict())
                wandb.finish()

            if self.dist and dist.is_initialized():
                dist.barrier()
        except Exception as e:
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
        
        loss_record = self.evaluator.init_record(["{}_loss".format(split)])
        all_y = []
        all_y_pred = []
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            for x, y in eval_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                y_pred = self.inference(x, y, **kwargs)
                # normalizer 可能是 dict {'hr': ..., 'lr': ...} 或单个对象
                _norm = self.normalizer['hr'] if isinstance(self.normalizer, dict) else self.normalizer
                y_pred = _norm.decode(y_pred)
                y = _norm.decode(y)
                all_y.append(y)
                all_y_pred.append(y_pred)
        y = torch.cat(all_y, dim=0)
        y_pred = torch.cat(all_y_pred, dim=0)
        loss = self.loss_fn(y_pred, y, mask=self.mask_hr)
        total_samples = y.size(0)
        loss_record.update({"{}_loss".format(split): loss.item()}, n=total_samples)
        self.evaluator(y_pred, y, record=loss_record, mask=self.mask_hr)
        if self.dist and dist.is_initialized():
            loss_record.dist_reduce()
        return loss_record
