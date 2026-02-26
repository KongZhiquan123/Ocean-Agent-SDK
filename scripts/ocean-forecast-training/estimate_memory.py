"""
@file estimate_memory.py
@description GPU memory estimation for ocean forecast training via dry-run.
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training

Usage:
    python estimate_memory.py --config /path/to/config.yaml [--device 0]

Output JSON:
    {
        "status": "success" | "oom" | "error",
        "gpu_name": "NVIDIA A100",
        "gpu_total_mb": 40960,
        "model_params_mb": 12.5,
        "peak_memory_mb": 3200,
        "available_mb": 40960,
        "utilization_pct": 7.8,
        "batch_size": 4,
        "use_amp": false,
        "gradient_checkpointing": false,
        "recommendations": ["..."]
    }
"""

import os
import sys
import json
import yaml
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def estimate_memory(config_path, device_id=0):
    with open(config_path, 'r') as f:
        args = yaml.load(f, yaml.FullLoader)

    model_name = args['model']['name']
    batch_size = args['data'].get('train_batchsize', 4)
    use_amp = args['train'].get('use_amp', False)
    gradient_checkpointing = args['train'].get('gradient_checkpointing', False)
    spatial_shape = args['data']['shape']  # [H, W]
    in_channels = args['model'].get('in_channels', 1)
    out_channels = args['model'].get('out_channels', 1)

    if not torch.cuda.is_available():
        return {
            "status": "error",
            "message": "CUDA not available",
            "recommendations": ["Please use a machine with NVIDIA GPU"],
        }

    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_total_bytes = torch.cuda.get_device_properties(device_id).total_memory
    gpu_total_mb = gpu_total_bytes / (1024 ** 2)

    from models import MODEL_REGISTRY

    try:
        model = MODEL_REGISTRY[model_name](args['model'])
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to build model: {str(e)}",
            "recommendations": ["Check model configuration"],
        }

    model = model.to(device)
    model_params = sum(p.numel() for p in model.parameters())
    model_params_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)

    H, W = spatial_shape[0], spatial_shape[1]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_id)

    result = {
        "gpu_name": gpu_name,
        "gpu_total_mb": round(gpu_total_mb, 1),
        "model_name": model_name,
        "model_params": model_params,
        "model_params_mb": round(model_params_mb, 2),
        "batch_size": batch_size,
        "input_shape": [batch_size, H, W, in_channels],
        "output_shape": [batch_size, H, W, out_channels],
        "use_amp": use_amp,
        "gradient_checkpointing": gradient_checkpointing,
    }

    try:
        model.train()

        x = torch.randn(batch_size, H, W, in_channels, device=device)
        y = torch.randn(batch_size, H, W, out_channels, device=device)

        with torch.amp.autocast('cuda', enabled=use_amp):
            if gradient_checkpointing:
                from torch.utils import checkpoint
                y_pred = checkpoint.checkpoint(model, x, use_reentrant=False)
            else:
                y_pred = model(x)
            loss = ((y_pred - y) ** 2).mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        peak_bytes = torch.cuda.max_memory_allocated(device_id)
        peak_mb = peak_bytes / (1024 ** 2)

        result["status"] = "success"
        result["peak_memory_mb"] = round(peak_mb, 1)
        result["available_mb"] = round(gpu_total_mb, 1)
        result["utilization_pct"] = round(peak_mb / gpu_total_mb * 100, 1)
        result["recommendations"] = _generate_recommendations(result)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        result["status"] = "oom"
        result["peak_memory_mb"] = round(gpu_total_mb, 1)
        result["available_mb"] = round(gpu_total_mb, 1)
        result["utilization_pct"] = 100.0
        result["recommendations"] = _generate_oom_recommendations(result)

    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        result["recommendations"] = []

    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    return result


def _generate_recommendations(result):
    recs = []
    pct = result["utilization_pct"]

    if pct > 90:
        recs.append(f"显存使用率 {pct}% 接近上限，训练中可能因波动 OOM")
        if not result["use_amp"]:
            recs.append("建议: 启用 use_amp=true 可减少约 40-50% 显存")
        if not result["gradient_checkpointing"]:
            recs.append("建议: 启用 gradient_checkpointing=true 可减少约 60% 激活显存")
        recs.append(f"建议: 减小 batch_size (当前 {result['batch_size']})")
    elif pct > 75:
        recs.append(f"显存使用率 {pct}%，余量较小但可以训练")
        if not result["use_amp"]:
            recs.append("建议: 启用 use_amp=true 可进一步降低显存")
    elif pct > 50:
        recs.append(f"显存使用率 {pct}%，余量充足")
        recs.append("可尝试增大 batch_size 以提高训练速度")
    else:
        recs.append(f"显存使用率 {pct}%，余量很充足")
        recs.append("可显著增大 batch_size 或使用更大的模型")

    return recs


def _generate_oom_recommendations(result):
    recs = [f"当前配置 OOM！GPU: {result['gpu_name']} ({result['gpu_total_mb']:.0f} MB)"]

    if not result["use_amp"]:
        recs.append("1. 启用 use_amp=true（减少约 40-50% 显存）")
    if not result["gradient_checkpointing"]:
        recs.append("2. 启用 gradient_checkpointing=true（减少约 60% 激活显存）")

    bs = result["batch_size"]
    if bs > 1:
        recs.append(f"3. 减小 batch_size: {bs} -> {max(bs // 2, 1)}")

    return recs


def main():
    parser = argparse.ArgumentParser(description='Estimate GPU memory for forecast training')
    parser.add_argument('--config', type=str, required=True, help='Training config YAML path')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    result = estimate_memory(args.config, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
