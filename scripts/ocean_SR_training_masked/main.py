"""
main.py - 单卡训练入口

@author Leizheng
@contributors Leizheng
@date 2026-02-07
@version 1.2.0

@changelog
  - 2026-02-07 Leizheng: v1.2.0 始终输出错误事件（不限 startup 阶段）
  - 2026-02-07 Leizheng: v1.1.0 添加顶层 try-catch，结构化错误输出
  - 原始版本: v1.0.0
"""

import sys
import json
import traceback

from config import parser
from utils.helper import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config
from trainers import _trainer_dict
from models import _model_dict
from datasets import _dataset_dict


def main():
    trainer = None
    try:
        args = parser.parse_args()
        args = vars(args)
        args = load_config(args)

        saving_path, saving_name = set_up_logger(args)

        set_seed(args['train'].get('seed', 42))
        args['train']['saving_path'] = saving_path
        args['train']['saving_name'] = saving_name
        save_config(args, saving_path)

        model_name = args['model']['name']
        trainer = _trainer_dict[model_name](args)
        trainer.process()
    except Exception as e:
        # 始终输出结构化错误事件
        # base.py 的 process() 可能已经输出过 training_error（带 epoch 等详细信息），
        # 这里作为兜底（尤其是 startup 阶段 trainer 还没创建的情况）
        error = {
            "event": "training_error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "phase": "startup" if trainer is None else "training",
        }
        print(f"__event__{json.dumps(error, ensure_ascii=False)}__event__", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
