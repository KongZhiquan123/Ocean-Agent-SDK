"""
@file generate_config.py
@description Generate YAML training configuration for ocean forecast models.
             Replaces scale/patch_size with in_t/out_t/stride for time-series forecasting.
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-27 Leizheng: v1.1.0 add 10 NeuralFramework models support
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training

Usage:
    python generate_config.py --params '<JSON string>' --output /path/to/config.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml


# Model default hyperparameters
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
    "SwinTransformerV2": {},
    "SwinMLP": {},
    "Transformer": {},
    "M2NO2d": {},
    # NeuralFramework models
    "OceanCNN": {},
    "OceanResNet": {},
    "OceanViT": {"patch_size": 8, "d_model": 256, "nhead": 8, "num_layers": 6},
    "Fuxi": {"embed_dim": 192, "num_groups": 32, "num_heads": 6, "window_size": 7, "depth": 8, "use_3d_path": False},
    "Fengwu": {"embed_dim": 192, "num_heads": [6, 12, 12, 6], "window_size": [6, 6], "depth": 6},
    "Pangu": {"embed_dim": 192, "num_heads": [6, 12, 12, 6], "window_size": [2, 6, 6], "depth": 6},
    "Crossformer": {"d_model": 256, "n_heads": 4, "d_ff": 512, "seg_len": 6, "e_layers": 3, "d_layers": 1},
    "NNG": {"hidden_dim": 256, "num_processor_layers": 8, "mesh_level": 3},
    "OneForecast": {"hidden_dim": 256, "num_processor_layers": 8, "mesh_level": 3},
    "GraphCast": {"hidden_dim": 256, "num_processor_layers": 8, "mesh_level": 3},
}

# Models that don't need scale/upsample (forecast = same resolution in/out)
FORECAST_MODELS = {
    "FNO2d", "UNet2d", "UNet3d", "FNO3d", "Transformer",
    "SwinTransformerV2", "SwinMLP", "M2NO2d", "GalerkinTransformer",
    "Transolver", "GNOT", "ONO", "LSM", "LNO", "MLP", "UNet1d", "FNO1d",
    "OceanCNN", "OceanResNet", "OceanViT",
    "Fuxi", "Fengwu", "Pangu", "Crossformer",
    "NNG", "OneForecast", "GraphCast",
}

AMP_AUTO_DISABLE_MODELS = {"FNO2d", "HiNOTE", "MWT2d", "M2NO2d", "MG-DDPM", "SRNO"}
HEAVY_MODELS = {
    "GalerkinTransformer", "MWT2d", "SRNO",
    "SwinTransformerV2", "SwinMLP",
    "Fuxi", "Fengwu", "Pangu",
    "NNG", "OneForecast", "GraphCast",
}

# NeuralFramework models that need in_t/out_t injected into model config
NF_MODELS = {
    "OceanCNN", "OceanResNet", "OceanViT",
    "Fuxi", "Fengwu", "Pangu", "Crossformer",
    "NNG", "OneForecast", "GraphCast",
}


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def detect_spatial_shape(dataset_root, dyn_vars):
    """Auto-detect spatial shape from first NPY file."""
    for split in ["train", "valid", "test"]:
        if dyn_vars:
            var_dir = os.path.join(dataset_root, split, dyn_vars[0])
            if os.path.isdir(var_dir):
                files = sorted([f for f in os.listdir(var_dir) if f.endswith(".npy")])
                if files:
                    arr = np.load(os.path.join(var_dir, files[0]))
                    return list(arr.shape)  # [H, W]
    return None


def generate_config(params):
    """
    Generate training configuration for ocean forecast.

    Required params:
        model_name (str): Model name
        dataset_root (str): Preprocessed data root directory
        dyn_vars (list[str]): Dynamic variable names
        log_dir (str): Log output directory

    Time-series specific:
        in_t (int): Input timesteps (default 7)
        out_t (int): Output timesteps (default 1)
        stride (int): Sliding window stride (default 1)

    Training params:
        epochs, lr, batch_size, etc. (same as SR version)
    """
    model_name = params["model_name"]
    dataset_root = params["dataset_root"]
    dyn_vars = params.get("dyn_vars", [])
    log_dir = params["log_dir"]

    # Time-series parameters
    in_t = params.get("in_t", 7)
    out_t = params.get("out_t", 1)
    stride = params.get("stride", 1)

    n_vars = len(dyn_vars)
    in_channels = in_t * n_vars
    out_channels = out_t * n_vars

    # Detect spatial shape
    spatial_shape = params.get("spatial_shape", None)
    if spatial_shape is None:
        # Try reading from var_names.json
        var_names_path = os.path.join(dataset_root, "var_names.json")
        if os.path.isfile(var_names_path):
            with open(var_names_path, "r") as f:
                var_info = json.load(f)
                spatial_shape = var_info.get("spatial_shape", None)

    if spatial_shape is None:
        spatial_shape = detect_spatial_shape(dataset_root, dyn_vars)

    if spatial_shape is None:
        raise ValueError("Cannot detect spatial shape. Please provide spatial_shape parameter.")

    # If dyn_vars not provided, try var_names.json
    if not dyn_vars:
        var_names_path = os.path.join(dataset_root, "var_names.json")
        if os.path.isfile(var_names_path):
            with open(var_names_path, "r") as f:
                var_info = json.load(f)
                dyn_vars = var_info.get("dynamic", var_info.get("dyn_vars", []))
                n_vars = len(dyn_vars)
                in_channels = in_t * n_vars
                out_channels = out_t * n_vars

    # Training params with defaults
    epochs = params.get("epochs", 500)
    lr = params.get("lr", 0.001)
    batch_size = params.get("batch_size", 4)
    eval_batch_size = params.get("eval_batch_size", 4)
    patience = params.get("patience", 10)
    eval_freq = params.get("eval_freq", 5)
    normalize = params.get("normalize", True)
    normalizer_type = params.get("normalizer_type", "PGN")
    optimizer = params.get("optimizer", "AdamW")
    weight_decay = params.get("weight_decay", 0.001)
    scheduler = params.get("scheduler", "StepLR")
    scheduler_step_size = params.get("scheduler_step_size", 300)
    scheduler_gamma = params.get("scheduler_gamma", 0.5)
    seed = params.get("seed", 42)
    wandb_enabled = params.get("wandb", False)
    device = params.get("device", 0)
    device_ids = params.get("device_ids", [0])
    distribute = params.get("distribute", False)
    distribute_mode = params.get("distribute_mode", "single")
    master_port = params.get("master_port", 29500)
    ckpt_path = params.get("ckpt_path", None)
    use_amp = params.get("use_amp", True)
    gradient_checkpointing = params.get("gradient_checkpointing", True)

    # AMP auto-disable for FFT models
    if model_name in AMP_AUTO_DISABLE_MODELS and use_amp:
        if not params.get("_user_specified_amp", False):
            use_amp = False

    # Build model config
    model_config = {
        "name": model_name,
        "in_channels": in_channels,
        "out_channels": out_channels,
    }

    # Merge model defaults
    if model_name in MODEL_DEFAULTS:
        model_config.update(MODEL_DEFAULTS[model_name])

    # NeuralFramework models: inject in_t/out_t and spatial dimensions
    if model_name in NF_MODELS:
        model_config["in_t"] = in_t
        model_config["out_t"] = out_t
        if model_name in {"Fuxi", "Fengwu", "Pangu", "Crossformer"}:
            model_config.setdefault("img_size", spatial_shape)
        if model_name in {"NNG", "OneForecast", "GraphCast"}:
            model_config.setdefault("input_res", spatial_shape)

    # FNO2d specific: no upsample for forecast (same resolution in/out)
    if model_name == "FNO2d":
        model_config["in_dim"] = in_channels
        model_config["out_dim"] = out_channels
        model_config["upsample_factor"] = [1, 1]  # No upsampling for forecast

    # Galerkin/other models that need explicit dims
    if model_name in {"GalerkinTransformer", "MWT2d", "SRNO"}:
        model_config["upsample_factor"] = [1, 1]
        if model_name == "GalerkinTransformer":
            model_config["in_dim"] = in_channels
            model_config["out_dim"] = out_channels

    # Build data config
    data_config = {
        "name": "ocean_forecast_npy",
        "data_path": os.path.abspath(dataset_root),
        "dyn_vars": dyn_vars,
        "in_t": in_t,
        "out_t": out_t,
        "stride": stride,
        "normalize": normalize,
        "normalizer_type": normalizer_type,
        "shape": spatial_shape,
        "train_batchsize": batch_size,
        "eval_batchsize": eval_batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }

    # Build train config
    train_config = {
        "epochs": epochs,
        "eval_freq": eval_freq,
        "patience": patience,
        "cuda": True,
        "device_ids": device_ids,
        "seed": seed,
        "saving_best": True,
        "saving_ckpt": False,
        "distribute_mode": distribute_mode,
        "use_amp": use_amp,
        "gradient_checkpointing": gradient_checkpointing,
    }

    if ckpt_path:
        train_config["load_ckpt"] = True
        train_config["ckpt_path"] = ckpt_path

    # Build optimizer config
    optim_config = {
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay,
    }

    # Build scheduler config
    schedule_config = {
        "scheduler": scheduler,
    }
    if scheduler == "StepLR":
        schedule_config["step_size"] = scheduler_step_size
        schedule_config["gamma"] = scheduler_gamma
    elif scheduler == "MultiStepLR":
        milestones = params.get("milestones", [100, 200, 300])
        schedule_config["milestones"] = milestones
        schedule_config["gamma"] = scheduler_gamma
    elif scheduler == "OneCycleLR":
        schedule_config["div_factor"] = params.get("div_factor", 25)
        schedule_config["final_div_factor"] = params.get("final_div_factor", 10000)
        schedule_config["pct_start"] = params.get("pct_start", 0.3)
        # Calculate steps_per_epoch from data
        schedule_config["steps_per_epoch"] = 100  # will be overridden at runtime

    # Build log config
    log_config = {
        "log": True,
        "log_dir": log_dir,
        "saving_path": log_dir,
        "wandb": wandb_enabled,
        "wandb_project": params.get("wandb_project", "ocean-forecast"),
    }

    # Build evaluate config (forecast uses RMSE/MAE as primary metrics)
    evaluate_config = {
        "metrics": [
            {"name": "rmse", "key": "rmse"},
            {"name": "mae", "key": "mae"},
            {"name": "mse", "key": "mse"},
        ],
        "strict": False,
        "kwargs": {},
    }

    if spatial_shape:
        evaluate_config["kwargs"]["shape"] = spatial_shape

    # Build loss config (default: LpLoss)
    loss_config = params.get("loss", None)

    # Assemble full config
    config = {
        "model": model_config,
        "data": data_config,
        "train": train_config,
        "optimize": optim_config,
        "schedule": schedule_config,
        "log": log_config,
        "evaluate": evaluate_config,
    }

    if loss_config:
        config["loss"] = loss_config

    return config


def main():
    parser = argparse.ArgumentParser(description="Generate forecast training config")
    parser.add_argument("--params", required=True, help="JSON string of parameters")
    parser.add_argument("--output", required=True, help="Output YAML path")
    args = parser.parse_args()

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "error": f"JSON parse error: {e}"}))
        sys.exit(1)

    try:
        config = generate_config(params)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

    # Write YAML
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    result = {
        "status": "success",
        "config_path": output_path,
        "model_name": params.get("model_name"),
        "in_channels": config["model"]["in_channels"],
        "out_channels": config["model"]["out_channels"],
        "in_t": config["data"]["in_t"],
        "out_t": config["data"]["out_t"],
        "spatial_shape": config["data"].get("shape"),
    }

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
