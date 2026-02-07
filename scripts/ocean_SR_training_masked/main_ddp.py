import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import parser
from utils.helper import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config, save_code
from trainers import _trainer_dict
from models import _model_dict
from datasets import _dataset_dict


def main():
    # ============ Step 1. 初始化分布式环境 ============
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ============ Step 2. 解析参数与加载配置 ============
    args = parser.parse_args()
    args = vars(args)
    args = load_config(args)
    args['train']['local_rank'] = local_rank
    args['train']['world_size'] = dist.get_world_size()
    args['train']['rank'] = dist.get_rank()

    # ============ Step 3. 只在 rank=0 初始化日志与保存配置 ============
    if dist.get_rank() == 0:
        saving_path, saving_name = set_up_logger(args)
        save_config(args, saving_path)

        # 保存本次训练用到的相关代码快照
        model_name = args['model']['name']
        data_name = args['data']['name']
        if model_name in _model_dict:
            model_entry = _model_dict[model_name]
            if isinstance(model_entry, dict):
                save_code(model_entry['model'], saving_path, with_dir=True)
            else:
                save_code(model_entry, saving_path, with_dir=True)
        if data_name in _dataset_dict:
            save_code(_dataset_dict[data_name], saving_path)
        if model_name in _trainer_dict:
            save_code(_trainer_dict[model_name], saving_path)
        import utils.normalizer, utils.loss
        save_code(utils.normalizer, saving_path)
        save_code(utils.loss, saving_path)
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

    # ============ Step 7. 关闭分布式环境 ============
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
