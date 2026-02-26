"""
main_ddp.py - 多卡 DDP 训练入口（海洋时序预测）

@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
    - Based on SR main_ddp.py, adapted for forecast training registries
    - Uses TRAINER_REGISTRY instead of _trainer_dict
    - Supports train / predict modes
"""

import os
import sys
import json
import traceback
import torch
import torch.distributed as dist

from config import parser
from utils.helper import set_up_logger, set_seed, set_device, load_config, save_config
from trainers import TRAINER_REGISTRY


def _log_json_event(event_data):
    """Emit structured JSON event for Agent process manager."""
    json_str = json.dumps(event_data, ensure_ascii=False)
    print(f"__event__{json_str}__event__", flush=True)


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

        train_cfg = args.setdefault("train", {})
        train_cfg["local_rank"] = local_rank
        train_cfg["world_size"] = dist.get_world_size()
        train_cfg["rank"] = rank

        # ============ Step 3. 只在 rank=0 初始化日志与保存配置 ============
        if rank == 0:
            saving_path, saving_name = set_up_logger(args)
            save_config(args, saving_path)
        else:
            saving_path, saving_name = None, None

        payload = [saving_path, saving_name]
        dist.broadcast_object_list(payload, src=0)
        saving_path, saving_name = payload

        train_cfg["saving_path"] = saving_path
        train_cfg["saving_name"] = saving_name

        # ============ Step 4. 固定随机种子与设备 ============
        set_seed(train_cfg.get("seed", 42))
        torch.cuda.set_device(local_rank)

        # ============ Step 5. 构建 trainer ============
        mode = args.get("mode", train_cfg.get("mode", "train"))

        # predict 模式不需要 __init__ 中的 load_ckpt
        if mode == "predict":
            train_cfg["load_ckpt"] = False

        trainer_name = args["model"]["name"]
        trainer_cls = TRAINER_REGISTRY[trainer_name]
        trainer = trainer_cls(args)

        # ============ Step 6. 根据 mode 执行对应流程 ============
        if mode == "predict":
            ckpt_path = train_cfg.get("ckpt_path", "") or os.path.join(
                saving_path, "best_model.pth"
            )
            trainer.load_model(ckpt_path)
            trainer.predict()
        else:
            trainer.process()

    except Exception as e:
        _log_json_event({
            "type": "training_error",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
            "phase": "startup" if trainer is None else "training",
        })
        traceback.print_exc()
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
