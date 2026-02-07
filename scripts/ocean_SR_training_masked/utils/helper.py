import os
import yaml
import torch
import logging
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def set_device(cuda, device):
    if cuda is True and torch.cuda.is_available():
        torch.cuda.set_device(device=device)


def load_config(args):
    with open(args['config'], 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    for k, v in config.items():
        args[k] = v
    return args


def save_config(args, saving_path):
    with open(os.path.join(saving_path, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)


def get_dir_path(args, create_dir=True):
    """直接使用 log_dir 作为输出目录，不再创建子目录"""
    path = args['log']['log_dir']
    if create_dir:
        os.makedirs(path, exist_ok=True)
    return path, os.path.basename(path)


def set_up_logger(args):
    log_dir, dir_name = get_dir_path(args)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(log_dir, "train.log"),
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(log_dir))

    return log_dir, dir_name
