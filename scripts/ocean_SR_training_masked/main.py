from config import parser
from utils.helper import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config, save_code
from trainers import _trainer_dict
from models import _model_dict
from datasets import _dataset_dict


def main():
    args = parser.parse_args()
    args = vars(args)
    args = load_config(args)

    saving_path, saving_name = set_up_logger(args)

    set_seed(args['train'].get('seed', 42))
    args['train']['saving_path'] = saving_path
    args['train']['saving_name'] = saving_name
    save_config(args, saving_path)

    # 保存本次训练用到的相关代码快照
    model_name = args['model']['name']
    data_name = args['data']['name']
    # 模型代码
    if model_name in _model_dict:
        model_entry = _model_dict[model_name]
        if isinstance(model_entry, dict):
            # 扩散模型：保存 model 和 diffusion 所在的包目录
            save_code(model_entry['model'], saving_path, with_dir=True)
        else:
            save_code(model_entry, saving_path, with_dir=True)
    # 数据集代码
    if data_name in _dataset_dict:
        save_code(_dataset_dict[data_name], saving_path)
    # trainer 代码
    if model_name in _trainer_dict:
        save_code(_trainer_dict[model_name], saving_path)
    # utils（normalizer, loss）
    import utils.normalizer, utils.loss
    save_code(utils.normalizer, saving_path)
    save_code(utils.loss, saving_path)

    trainer = _trainer_dict[model_name](args)
    trainer.process()


if __name__ == "__main__":
    main()
