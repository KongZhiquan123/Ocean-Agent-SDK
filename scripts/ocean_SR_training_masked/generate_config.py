"""
generate_config.py - 根据参数生成训练配置 YAML 文件

接收 JSON 格式参数，生成适配 OceanNPY 数据集的 YAML 配置文件。

用法:
    python generate_config.py --params '<JSON string>' --output /path/to/config.yaml

@author Leizheng
@contributors kongzhiquan
@date 2026-02-06
@version 3.1.0

@changelog
  - 2026-02-07 kongzhiquan: v3.1.0 use_amp 默认值改为 True（OOM 防护增强）
  - 2026-02-07 kongzhiquan: v3.0.0 新增 compute_model_divisor() 自动对齐 image_size
    - 根据模型架构计算输入尺寸的整除要求
    - 扩散模型自动向上对齐 image_size（如 400→416）
    - 将 model_divisor 写入 data config 供 OceanNPY 使用
  - 2026-02-07 Leizheng: v2.0.0 支持 OOM 防护参数
    - 新增 use_amp, gradient_checkpointing, patch_size 参数
    - 写入 train / data section 供 trainer 和 dataset 读取
  - 原始版本: v1.0.0
"""

import argparse
import json
import yaml
import sys
import os


# 模型默认参数（覆盖不了的就用模板）
MODEL_DEFAULTS = {
    "FNO2d": {
        "modes1": [15, 12, 9, 9, 9],
        "modes2": [15, 12, 9, 9, 9],
        "width": 64,
        "fc_dim": 128,
        "layers": [16, 24, 24, 32, 32],
        "act": "gelu",
        "pos_dim": 2,
    },
    "UNet2d": {},
    "SwinIR": {
        "patch_size": 1,
        "embed_dim": 120,
        "depths": [6, 6, 6, 6],
        "num_heads": [6, 6, 6, 6],
        "window_size": 8,
    },
    "EDSR": {},
    "HiNOTE": {},
    "DDPM": {
        "inner_channel": 64,
        "channel_mults": [1, 1, 2, 2, 4, 4],
        "attn_res": [16],
        "res_blocks": 2,
        "dropout": 0.2,
        "conditional": True,
        "n_iter": 10000,
        "loss_type": "lploss",
    },
    "SR3": {
        "inner_channel": 64,
        "channel_mults": [1, 1, 2, 2, 4, 4],
        "attn_res": [16],
        "res_blocks": 2,
        "dropout": 0.2,
        "conditional": True,
        "n_iter": 10000,
        "loss_type": "lploss",
    },
}

DIFFUSION_MODELS = {"DDPM", "SR3", "MG-DDPM", "ReMiG"}


def compute_model_divisor(model_name: str, model_config: dict) -> int:
    """根据模型架构计算输入尺寸的整除要求。

    Returns:
        divisor (int): 输入空间尺寸必须能被此值整除。
            - DDPM/SR3/ReMiG/ResShift: 2^(len(channel_mults)-1)，通常 = 32
            - UNet2d: 16（4 次 MaxPool）
            - SwinIR/FNO2d/EDSR/HiNOTE/M2NO2d: 1（无约束）
    """
    DIFFUSION_LIKE = {"ResShift"}

    if model_name in DIFFUSION_MODELS or model_name in DIFFUSION_LIKE:
        channel_mults = model_config.get("channel_mults", [1, 1, 2, 2, 4, 4])
        num_downsamples = len(channel_mults) - 1
        return 2 ** num_downsamples  # 通常 = 32
    elif model_name == "UNet2d":
        return 16  # 4 次 MaxPool (2^4)
    else:
        return 1  # FNO2d, EDSR, SwinIR, HiNOTE, M2NO2d 无约束


def generate_config(params):
    """
    根据参数生成训练配置。

    必需参数:
        model_name (str): 模型名称
        dataset_root (str): 预处理数据根目录
        dyn_vars (list[str]): 动态变量列表
        scale (int): 超分辨率倍数
        log_dir (str): 日志输出目录

    可选参数:
        epochs (int): 训练轮数 (默认 500)
        lr (float): 学习率 (默认 0.001)
        batch_size (int): batch size (默认 32)
        eval_batch_size (int): 评估 batch size (默认 32)
        device (int): 主 GPU 设备号 (默认 0)
        device_ids (list[int]): 多卡时使用的 GPU 列表，如 [0,1,2,3]
        distribute (bool): 是否启用多卡训练 (默认 False)
        distribute_mode (str): 多卡模式 'DP' 或 'DDP' (默认 'DDP')
        patience (int): 早停耐心值 (默认 10)
        eval_freq (int): 评估频率 (默认 5)
        normalize (bool): 是否归一化 (默认 True)
        normalizer_type (str): 归一化类型 (默认 'PGN')
        optimizer (str): 优化器 (默认 'AdamW')
        weight_decay (float): 权重衰减 (默认 0.001)
        scheduler (str): 调度器 (默认 'StepLR')
        scheduler_step_size (int): 调度器步长 (默认 300)
        scheduler_gamma (float): 调度器衰减率 (默认 0.5)
        seed (int): 随机种子 (默认 42)
        hr_shape (list[int]): HR 尺寸 [H, W] (若不提供则自动检测)
        use_amp (bool): 是否启用 AMP 混合精度 (默认 True)
        gradient_checkpointing (bool): 是否启用梯度检查点 (默认 False)
        patch_size (int): Patch 裁剪尺寸，None 表示全图训练 (默认 None)
    """
    model_name = params['model_name']
    dataset_root = params['dataset_root']
    dyn_vars = params['dyn_vars']
    scale = params['scale']
    log_dir = params['log_dir']

    n_channels = len(dyn_vars)
    lr_size = None
    hr_shape = params.get('hr_shape', None)

    # 自动检测 HR shape
    if hr_shape is None:
        import numpy as np
        sample_dir = os.path.join(dataset_root, 'train', 'hr', dyn_vars[0])
        if os.path.isdir(sample_dir):
            npy_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.npy')])
            if npy_files:
                sample = np.load(os.path.join(sample_dir, npy_files[0]))
                hr_shape = list(sample.shape)  # [H, W]

    if hr_shape is None:
        raise ValueError("Cannot detect HR shape. Please provide hr_shape parameter.")

    lr_size = hr_shape[0] // scale

    # 构建 model config
    model_config = {
        "name": model_name,
        "in_channels": n_channels,
        "out_channels": n_channels,
    }

    # 合并模型默认参数
    if model_name in MODEL_DEFAULTS:
        model_config.update(MODEL_DEFAULTS[model_name])

    # 特定模型参数
    if model_name == "FNO2d":
        model_config["in_dim"] = n_channels
        model_config["out_dim"] = n_channels
        model_config["upsample_factor"] = [scale, scale]
    elif model_name == "SwinIR":
        model_config["img_size"] = lr_size
        model_config["upscale_factor"] = scale
    elif model_name in DIFFUSION_MODELS:
        model_config["in_channel"] = n_channels * 2  # LR + noise
        model_config["out_channel"] = n_channels
        # 扩散模型以 patch 为单位处理: 有 patch_size 时以 patch 为基准，否则用全图
        if patch_size:
            model_config["image_size"] = patch_size
        else:
            model_config["image_size"] = hr_shape[0]

    # 计算模型整除要求并自动对齐 image_size
    divisor = compute_model_divisor(model_name, model_config)
    if divisor > 1:
        # 基准尺寸: 有 patch_size 时用 patch_size，否则用全图
        base_size = patch_size if (patch_size and model_name in DIFFUSION_MODELS) else hr_shape[0]
        raw_size = base_size
        aligned_size = ((raw_size + divisor - 1) // divisor) * divisor
        if aligned_size != raw_size:
            print(f"[generate_config] 自动对齐 image_size: {raw_size} -> {aligned_size} "
                  f"(模型 {model_name} 要求被 {divisor} 整除)", file=sys.stderr)
        if model_name in DIFFUSION_MODELS:
            model_config["image_size"] = aligned_size
            model_config["raw_image_size"] = raw_size
            model_config["scale"] = scale

    # 构建完整配置
    config = {
        "model": model_config,
        "data": {
            "name": "OceanNPY",
            "dataset_root": dataset_root,
            "dyn_vars": dyn_vars,
            "sample_factor": scale,
            "shape": hr_shape,
            "train_batchsize": params.get("batch_size", 32),
            # 扩散模型验证需要完整采样循环（2000步），显存开销远大于训练
            # 默认 eval_batchsize 对扩散模型设为 4，非扩散模型保持 32
            "eval_batchsize": params.get("eval_batch_size",
                                         4 if model_name in DIFFUSION_MODELS else 32),
            "normalize": params.get("normalize", True),
            "normalizer_type": params.get("normalizer_type", "PGN"),
            "patch_size": params.get("patch_size", None),
            "model_divisor": divisor,
        },
        "train": {
            "epochs": params.get("epochs", 500),
            "patience": params.get("patience", 10),
            "eval_freq": params.get("eval_freq", 5),
            "cuda": True,
            "device": params.get("device", 0),
            "distribute": params.get("distribute", False),
            "distribute_mode": params.get("distribute_mode", "DDP"),
            "device_ids": params.get("device_ids", [0]),
            "seed": params.get("seed", 42),
            "saving_best": True,
            "saving_ckpt": params.get("saving_ckpt", False),
            "ckpt_freq": params.get("ckpt_freq", 100),
            "use_amp": params.get("use_amp", True),   # AMP 默认开启
            "gradient_checkpointing": params.get("gradient_checkpointing", False),
        },
        "optimize": {
            "optimizer": params.get("optimizer", "AdamW"),
            "lr": params.get("lr", 0.001),
            "weight_decay": params.get("weight_decay", 0.001),
        },
        "schedule": {
            "scheduler": params.get("scheduler", "StepLR"),
            "step_size": params.get("scheduler_step_size", 300),
            "gamma": params.get("scheduler_gamma", 0.5),
        },
        "log": {
            "log_dir": log_dir,
            "wandb": params.get("wandb", False),
            "wandb_project": params.get("wandb_project", f"OceanSR-{model_name}"),
        },
    }

    # 扩散模型额外配置
    if model_name in DIFFUSION_MODELS:
        config["beta_schedule"] = {
            "train": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-4,
                "linear_end": 2e-2,
            },
            "val": {
                "schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-4,
                "linear_end": 2e-2,
            },
        }

    return config


def main():
    parser = argparse.ArgumentParser(description='Generate training config YAML')
    parser.add_argument('--params', type=str, required=True, help='JSON string of parameters')
    parser.add_argument('--output', type=str, required=True, help='Output YAML file path')
    args = parser.parse_args()

    params = json.loads(args.params)
    config = generate_config(params)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(json.dumps({
        "status": "success",
        "config_path": os.path.abspath(args.output),
        "model": config["model"]["name"],
        "dataset": config["data"]["name"],
        "hr_shape": config["data"]["shape"],
        "epochs": config["train"]["epochs"],
        "distribute": config["train"]["distribute"],
        "distribute_mode": config["train"]["distribute_mode"],
        "device_ids": config["train"]["device_ids"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
