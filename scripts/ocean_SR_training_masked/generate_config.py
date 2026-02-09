"""
@file generate_config.py

@description 根据参数生成训练配置 YAML 文件
@author Leizheng
@date 2026-02-09
@version 3.8.5

@changelog
  - 2026-02-09 Leizheng: v3.8.5 扩散模型 eval_batchsize 限制为 <=4
  - 2026-02-09 Leizheng: v3.8.4 ReMiG 模板改为 remig.yaml
  - 2026-02-09 Leizheng: v3.8.3 FFT 模型 AMP 默认关闭
  - 2026-02-09 Leizheng: v3.8.2 gradient_checkpointing 默认开启
  - 2026-02-09 Leizheng: v3.8.1 默认 batch_size 下调为 4 + 默认开启 gradient_checkpointing
  - 2026-02-09 Leizheng: v3.8.0 修复 Galerkin/SRNO 多变量通道映射
  - 2026-02-09 Leizheng: v3.7.0 gradient_checkpointing 默认按模型/全图自适应
  - 2026-02-09 Leizheng: v3.6.0 默认 batch_size 下调为 16
  - 2026-02-09 Leizheng: v3.5.0 FNO 类模型默认关闭自动 patch
  - 2026-02-09 Leizheng: v3.4.0 补充标准模型 scale 参数映射
  - 2026-02-08 Leizheng: v3.3.0 自动写入 patch_size 并增加合法性校验
  - 2026-02-08 Leizheng: v3.2.0 新增 ckpt_path / load_ckpt 写入 train section
  - 2026-02-07 Leizheng: v3.1.0 use_amp 默认值改为 True（OOM 防护增强）
  - 2026-02-07 Leizheng: v3.0.0 新增 compute_model_divisor() 自动对齐 image_size
    - 根据模型架构计算输入尺寸的整除要求
    - 扩散模型自动向上对齐 image_size（如 400→416）
    - 将 model_divisor 写入 data config 供 OceanNPY 使用
  - 2026-02-07 Leizheng: v2.0.0 支持 OOM 防护参数
    - 新增 use_amp, gradient_checkpointing, patch_size 参数
    - 写入 train / data section 供 trainer 和 dataset 读取
  - 原始版本: v1.0.0

用法:
    python generate_config.py --params '<JSON string>' --output /path/to/config.yaml
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
RESSHIFT_MODELS = {"Resshift", "ResShift"}
# FNO/FFT 类模型默认不切 patch（显存充足且全图更稳定）
NO_PATCH_MODELS = {"FNO2d", "HiNOTE", "MWT2d", "M2NO2d"}
AMP_AUTO_DISABLE_MODELS = {"FNO2d", "HiNOTE", "MWT2d", "M2NO2d", "MG-DDPM"}
HEAVY_MODELS = {
    "Galerkin_Transformer",
    "MWT2d",
    "SRNO",
    "Swin_Transformer",
    "SwinIR",
    "DDPM",
    "SR3",
    "MG-DDPM",
    "Resshift",
    "ResShift",
    "ReMiG",
}

TEMPLATE_MAP = {
    "FNO2d": "fno.yaml",
    "UNet2d": "unet.yaml",
    "M2NO2d": "m2no.yaml",
    "Galerkin_Transformer": "galerkin.yaml",
    "MWT2d": "MWT.yaml",
    "SRNO": "sronet.yaml",
    "Swin_Transformer": "swin.yaml",
    "EDSR": "EDSR.yaml",
    "HiNOTE": "HiNOTE.yaml",
    "SwinIR": "swinIR.yaml",
    "DDPM": "ddpm.yaml",
    "SR3": "sr3.yaml",
    "MG-DDPM": "mg_ddpm.yaml",
    "Resshift": "resshift.yaml",
    "ResShift": "resshift.yaml",
    "ReMiG": "remig.yaml",
}


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_template_config(model_name: str):
    filename = TEMPLATE_MAP.get(model_name)
    if not filename:
        return None
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template_configs", "ns2d")
    template_path = os.path.join(template_dir, filename)
    if not os.path.isfile(template_path):
        return None
    with open(template_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_model_divisor(model_name: str, model_config: dict) -> int:
    """根据模型架构计算输入尺寸的整除要求。

    Returns:
        divisor (int): 输入空间尺寸必须能被此值整除。
            - DDPM/SR3/ReMiG/ResShift: 2^(len(channel_mults)-1)，通常 = 32
            - UNet2d: 16（4 次 MaxPool）
            - SwinIR/FNO2d/EDSR/HiNOTE/M2NO2d: 1（无约束）
    """
    DIFFUSION_LIKE = {"ResShift", "Resshift"}

    if model_name in DIFFUSION_MODELS or model_name in DIFFUSION_LIKE:
        channel_mults = model_config.get(
            "channel_mults",
            model_config.get("channel_mult", [1, 1, 2, 2, 4, 4])
        )
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
        batch_size (int): batch size (默认 4)
        eval_batch_size (int): 评估 batch size (默认 4)
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
        use_amp (bool): 是否启用 AMP 混合精度 (默认按模型: FFT 关闭)
        gradient_checkpointing (bool): 是否启用梯度检查点 (默认开启，可手动关闭)
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
    patch_size = params.get('patch_size', None)
    eval_batch_size = params.get('eval_batch_size', 4)
    try:
        eval_batch_size = int(eval_batch_size)
    except (TypeError, ValueError):
        eval_batch_size = 4
    if model_name in DIFFUSION_MODELS and eval_batch_size > 4:
        print(
            f"[generate_config] diffusion eval_batch_size={eval_batch_size} is too high; cap to 4 to reduce OOM risk",
            file=sys.stderr,
        )
        eval_batch_size = 4
    user_auto_patch = params.get('auto_patch', None)
    if user_auto_patch is None:
        auto_patch = False if (patch_size is None and model_name in NO_PATCH_MODELS) else True
    else:
        auto_patch = bool(user_auto_patch)

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

    # ResShift 默认 channel_mult（用于 divisor 计算）
    if model_name in RESSHIFT_MODELS:
        model_config["channel_mults"] = [1, 2, 2, 4]

    # 特定模型参数
    if model_name == "FNO2d":
        model_config["in_dim"] = n_channels
        model_config["out_dim"] = n_channels
        model_config["upsample_factor"] = [scale, scale]
    elif model_name in {"Galerkin_Transformer", "MWT2d", "Swin_Transformer", "SRNO"}:
        model_config["upsample_factor"] = [scale, scale]
        if model_name == "Galerkin_Transformer":
            model_config["in_dim"] = n_channels
            model_config["out_dim"] = n_channels
        elif model_name == "SRNO":
            model_config["input_channels"] = n_channels
            model_config["output_channels"] = n_channels
    elif model_name == "EDSR":
        model_config["upscale_factor"] = scale
    elif model_name == "SwinIR":
        model_config["img_size"] = lr_size
        model_config["upscale_factor"] = scale
    elif model_name == "HiNOTE":
        model_config["scale_factor"] = scale
    elif model_name in DIFFUSION_MODELS:
        model_config["in_channel"] = n_channels * 2  # LR + noise
        model_config["out_channel"] = n_channels

    # 计算模型整除要求并自动对齐 image_size
    divisor = compute_model_divisor(model_name, model_config)

    # 自动计算 patch_size（与 OceanNPY 保持一致）
    if patch_size is None and auto_patch:
        from math import gcd
        max_dim = min(hr_shape[0], hr_shape[1])
        lcm_factor = (scale * divisor) // gcd(scale, divisor)
        target = min(max_dim // 2, 256)
        auto_patch_size = (target // lcm_factor) * lcm_factor
        if auto_patch_size < lcm_factor and lcm_factor < max_dim:
            auto_patch_size = lcm_factor
        if 0 < auto_patch_size < max_dim:
            patch_size = int(auto_patch_size)
    elif patch_size is None and not auto_patch:
        print(f"[generate_config] 自动 patch 已关闭（模型 {model_name}），使用全图训练", file=sys.stderr)

    if patch_size is not None:
        max_dim = min(hr_shape[0], hr_shape[1])
        if patch_size <= 0 or patch_size > max_dim:
            raise ValueError(
                f"patch_size ({patch_size}) must be within (0, {max_dim}]"
            )
        if patch_size % scale != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by scale ({scale})"
            )
        if patch_size % divisor != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by model_divisor ({divisor})"
            )

    # 扩散模型以 patch 为单位处理: 有 patch_size 时以 patch 为基准，否则用全图
    if model_name in DIFFUSION_MODELS:
        if patch_size:
            model_config["image_size"] = patch_size
        else:
            model_config["image_size"] = hr_shape[0]

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

    # ResShift 配置（独立于 model_config）
    resshift_block = None
    if model_name in RESSHIFT_MODELS:
        image_size = patch_size if patch_size else hr_shape[0]
        resshift_block = {
            "model": {
                "target": "models.resshift.models.unet.UNetModelSwin",
                "ckpt_path": None,
                "params": {
                    "image_size": image_size,
                    "in_channels": n_channels,
                    "model_channels": 160,
                    "out_channels": n_channels,
                    "attention_resolutions": [64, 32, 16, 8],
                    "dropout": 0,
                    "channel_mult": [1, 2, 2, 4],
                    "num_res_blocks": [2, 2, 2, 2],
                    "conv_resample": True,
                    "dims": 2,
                    "use_fp16": False,
                    "num_head_channels": 32,
                    "use_scale_shift_norm": True,
                    "resblock_updown": False,
                    "swin_depth": 2,
                    "swin_embed_dim": 192,
                    "window_size": 8,
                    "mlp_ratio": 4,
                    "cond_lq": True,
                    "cond_mask": False,
                    "lq_size": image_size,
                },
            },
            "diffusion": {
                "target": "models.resshift.models.script_util.create_gaussian_diffusion",
                "params": {
                    "sf": scale,
                    "schedule_name": "exponential",
                    "schedule_kwargs": {
                        "power": 0.3,
                    },
                    "etas_end": 0.99,
                    "steps": 15,
                    "min_noise_level": 0.04,
                    "kappa": 2.0,
                    "weighted_mse": False,
                    "predict_type": "xstart",
                    "timestep_respacing": None,
                    "scale_factor": 1.0,
                    "normalize_input": False,
                    "latent_flag": False,
                },
            },
        }

    # gradient_checkpointing 默认开启（允许用户显式关闭）
    if "gradient_checkpointing" in params:
        gradient_checkpointing = bool(params.get("gradient_checkpointing"))
    else:
        gradient_checkpointing = True

    # use_amp 默认策略：FFT 模型关闭，其余开启（允许用户显式覆盖）
    if "use_amp" in params:
        use_amp = bool(params.get("use_amp"))
    else:
        use_amp = model_name not in AMP_AUTO_DISABLE_MODELS

    # 构建完整配置
    config = {
        "model": model_config,
        "data": {
            "name": "OceanNPY",
            "dataset_root": dataset_root,
            "dyn_vars": dyn_vars,
            "sample_factor": scale,
            "shape": hr_shape,
            "train_batchsize": params.get("batch_size", 4),
            # 扩散模型验证需要完整采样循环（2000步），显存开销远大于训练
            # 默认 eval_batchsize 设为 4，可按显存手动调整
            "eval_batchsize": eval_batch_size,
            "normalize": params.get("normalize", True),
            "normalizer_type": params.get("normalizer_type", "PGN"),
            "patch_size": patch_size,
            "auto_patch": auto_patch,
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
            "use_amp": use_amp,   # AMP 默认按模型策略
            "gradient_checkpointing": gradient_checkpointing,
            "load_ckpt": bool(params.get("ckpt_path")),
            "ckpt_path": params.get("ckpt_path", ""),
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

    if resshift_block is not None:
        config["resshift"] = resshift_block

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

    template_cfg = load_template_config(model_name)
    if template_cfg:
        config = deep_merge(template_cfg, config)

    if model_name in DIFFUSION_MODELS:
        try:
            final_eval_bs = int(config.get("data", {}).get("eval_batchsize", 4))
        except (TypeError, ValueError):
            final_eval_bs = 4
        if final_eval_bs > 4:
            print(
                f"[generate_config] diffusion eval_batchsize={final_eval_bs} after merge; cap to 4",
                file=sys.stderr,
            )
            config["data"]["eval_batchsize"] = 4

    return config


def main():
    parser = argparse.ArgumentParser(description='Generate training config YAML')
    parser.add_argument('--params', type=str, required=True, help='JSON string of parameters')
    parser.add_argument('--output', type=str, required=True, help='Output YAML file path')
    args = parser.parse_args()

    params = json.loads(args.params)
    model_name = params.get("model_name")
    requested_eval_batch = params.get("eval_batch_size", 4)
    try:
        requested_eval_batch = int(requested_eval_batch)
    except (TypeError, ValueError):
        requested_eval_batch = None
    config = generate_config(params)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(json.dumps({
        "status": "success",
        "eval_batchsize_requested": requested_eval_batch,
        "eval_batchsize_clamped": (
            model_name in DIFFUSION_MODELS
            and requested_eval_batch is not None
            and config["data"].get("eval_batchsize") is not None
            and config["data"].get("eval_batchsize") != requested_eval_batch
        ),
        "config_path": os.path.abspath(args.output),
        "model": config["model"]["name"],
        "dataset": config["data"]["name"],
        "hr_shape": config["data"]["shape"],
        "train_batchsize": config["data"].get("train_batchsize"),
        "eval_batchsize": config["data"].get("eval_batchsize"),
        "epochs": config["train"]["epochs"],
        "distribute": config["train"]["distribute"],
        "distribute_mode": config["train"]["distribute_mode"],
        "device_ids": config["train"]["device_ids"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
