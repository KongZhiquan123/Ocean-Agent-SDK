"""
main_ddp.py - 多卡 DDP 训练入口

@author Leizheng
@contributors kongzhiquan
@date 2026-02-07
@version 1.2.0

@changelog
  - 2026-02-07 kongzhiquan: v1.2.0 始终输出错误事件（不限 startup 阶段）
    - 训练阶段崩溃时 base.py 已输出详细 training_error，这里作为兜底
    - 区分 phase: startup / training，TypeScript 侧以最后收到的事件为准
  - 2026-02-07 kongzhiquan: v1.1.0 添加顶层 try-catch + rank 信息，finally 清理 dist
  - 原始版本: v1.0.0
"""

import os
import sys
import json
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import parser
from utils.helper import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config
from trainers import _trainer_dict
from models import _model_dict
from datasets import _dataset_dict


def main():
    rank = -1
    trainer = None
    try:
        # ============ Step 1. 初始化分布式环境 ============
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)

        # ============ Step 2. 解析参数与加载配置 ============
        args = parser.parse_args()
        args = vars(args)
        args = load_config(args)
        args['train']['local_rank'] = local_rank
        args['train']['world_size'] = dist.get_world_size()
        args['train']['rank'] = rank

        # ============ Step 3. 只在 rank=0 初始化日志与保存配置 ============
        if rank == 0:
            saving_path, saving_name = set_up_logger(args)
            save_config(args, saving_path)
        else:
            saving_path, saving_name = None, None

        payload = [saving_path, saving_name]
        dist.broadcast_object_list(payload, src=0)
        saving_path, saving_name = payload

        args['train']['saving_path'] = saving_path
        args['train']['saving_name'] = saving_name

        # ============ Step 4. 固定随机种子与设备 ============
        set_seed(args['train'].get('seed', 42))
        torch.cuda.set_device(local_rank)

        # ============ Step 5. 构建 trainer（内部会构建 model、dataloader） ============
        trainer = _trainer_dict[args['model']['name']](args)

        # ============ Step 6. 启动训练 ============
        trainer.process()
    except Exception as e:
        # 始终输出结构化错误事件，确保 TypeScript 进程管理器能捕获
        # base.py 的 process() 可能已经输出过 training_error（带 epoch 等详细信息），
        # 这里作为兜底（尤其是 startup 阶段 trainer 还没创建的情况）
        error = {
            "event": "training_error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
            "phase": "startup" if trainer is None else "training",
        }
        print(f"__event__{json.dumps(error, ensure_ascii=False)}__event__", flush=True)
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
