"""
@file main.py
@description Ocean forecast training entry point
             Registers OceanForecastNPYDataset and ForecastTrainer.
             Supports train / test / predict modes.
             Emits __event__ JSON markers for Agent process monitoring.
@author Leizheng
@date 2026-02-26
@version 2.0.0

@changelog
  - 2026-02-26 Leizheng: v2.0.0 forecast training support
    - Register OceanForecastNPYDataset
    - All models default to BaseTrainer (ForecastTrainer)
    - Add training_error event emission in except block
    - Support --mode train | test | predict
"""

import json
import sys
import traceback

import torch.distributed as dist

from config import parser
from utils import set_up_logger, set_seed, set_device, load_config, save_config, init_distributed
from trainers import TRAINER_REGISTRY


def _log_json_event(event_data):
    """Emit structured JSON event for Agent process manager."""
    # Normalise key: callers use "type" but TS expects "event"
    if "type" in event_data and "event" not in event_data:
        event_data["event"] = event_data.pop("type")
    json_str = json.dumps(event_data, ensure_ascii=False)
    print(f"__event__{json_str}__event__", flush=True)


def main():
    try:
        # ========= Step 0. Optional: initialize distributed =========
        distributed, local_rank = init_distributed()

        # ========= Step 1. Parse CLI args and load config =========
        args = parser.parse_args()
        args = vars(args)
        args = load_config(args)  # merge CLI args with YAML config

        # Ensure "train" sub-dict exists
        train_cfg = args.setdefault("train", {})

        if distributed:
            train_cfg["local_rank"] = local_rank
            train_cfg["world_size"] = dist.get_world_size()
            train_cfg["rank"] = dist.get_rank()
        else:
            train_cfg["local_rank"] = args['train'].get('device_ids', [0])[0]
            train_cfg["world_size"] = 1
            train_cfg["rank"] = args['train'].get('device_ids', [0])[0]

        # ========= Step 2. Setup logger and save config =========
        mode = args.get("mode", train_cfg.get("mode", "train"))

        if mode == "predict":
            # Predict mode: reuse existing saving_path (no new directory)
            saving_path = args.get("log", {}).get("saving_path") or args.get("log", {}).get("log_dir", ".")
            saving_name = "predict"
            import logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        elif distributed:
            # Only rank 0 creates logger and writes config to disk
            if train_cfg["rank"] == 0:
                saving_path, saving_name = set_up_logger(args)
                save_config(args, saving_path)
            else:
                saving_path, saving_name = None, None

            # Broadcast saving_path and saving_name to all ranks
            payload = [saving_path, saving_name]
            dist.broadcast_object_list(payload, src=0)
            saving_path, saving_name = payload
        else:
            # Single-process: just create logger and save config
            saving_path, saving_name = set_up_logger(args)
            save_config(args, saving_path)

        train_cfg["saving_path"] = saving_path
        train_cfg["saving_name"] = saving_name

        # ========= Step 3. Set random seed and (optionally) device =========
        seed = train_cfg.get("seed", 42)
        set_seed(seed)
        set_device(cuda=train_cfg.get("cuda", True), device=train_cfg["local_rank"])

        # ========= Step 4. Build trainer (creates model, dataloader, etc.) =========
        trainer_name = args["model"]["name"]
        trainer_cls = TRAINER_REGISTRY[trainer_name]
        trainer = trainer_cls(args)

        # ========= Step 5. Run training / evaluation / prediction =========
        if mode == "predict":
            trainer.predict()
        else:
            trainer.process()

        # ========= Step 6. Clean up distributed environment =========
        if distributed:
            dist.destroy_process_group()

    except Exception as e:
        # Emit training_error event for the Agent process manager
        _log_json_event({
            "type": "training_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        # Still print the traceback so it appears in logs
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
