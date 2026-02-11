"""
estimate_memory.py - 训练前 GPU 显存预估工具

通过 dry-run 一次 forward + backward 来测量实际峰值显存使用。
支持 AMP 混合精度和 gradient checkpointing 场景的估算。

用法:
    python estimate_memory.py --config /path/to/config.yaml [--device 0]

输出 JSON:
    {
        "status": "success" | "oom",
        "gpu_name": "NVIDIA A100",
        "gpu_total_mb": 40960,
        "model_params_mb": 12.5,
        "peak_memory_mb": 3200,
        "available_mb": 40960,
        "utilization_pct": 7.8,
        "batch_size": 16,
        "use_amp": false,
        "gradient_checkpointing": false,
        "patch_size": null,
        "recommendations": ["..."]
    }

@author Leizheng
@date 2026-02-07
@version 1.0.1

@changelog
  - 2026-02-11 Leizheng: v1.0.1 修复 torch 局部变量遮蔽
    - `import torch.utils.checkpoint` 改为 `from torch.utils import checkpoint`
    - 避免 Python 将 torch 视为局部变量导致 UnboundLocalError
  - 2026-02-07 Leizheng: v1.0.0 初始版本
    - dry-run forward + backward 测量峰值显存
    - 支持 AMP / gradient_checkpointing / patch_size 组合
    - 当 OOM 时自动给出建议的 batch_size / patch_size
"""

import os
import sys
import json
import yaml
import argparse
import torch
import numpy as np

# 添加父目录到 path，使 models/datasets/trainers 可以 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def estimate_memory(config_path, device_id=0):
    """
    通过 dry-run 测量 GPU 峰值显存。

    Args:
        config_path: YAML 配置文件路径
        device_id: GPU 设备号

    Returns:
        dict: 测量结果
    """
    with open(config_path, 'r') as f:
        args = yaml.load(f, yaml.FullLoader)

    model_name = args['model']['name']
    batch_size = args['data'].get('train_batchsize', 16)
    use_amp = args['train'].get('use_amp', False)
    gradient_checkpointing = args['train'].get('gradient_checkpointing', False)
    patch_size = args['data'].get('patch_size', None)
    hr_shape = args['data']['shape']  # [H, W]
    n_channels = len(args['data'].get('dyn_vars', ['temp']))
    scale = args['data'].get('sample_factor', 1)

    # GPU 信息
    if not torch.cuda.is_available():
        return {
            "status": "error",
            "message": "CUDA not available",
            "recommendations": ["Please use a machine with NVIDIA GPU"]
        }

    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_total_bytes = torch.cuda.get_device_properties(device_id).total_mem
    gpu_total_mb = gpu_total_bytes / (1024 ** 2)

    # 构建模型
    from models import _model_dict
    from models import _ddpm_dict

    DIFFUSION_MODELS = {"DDPM", "SR3", "MG-DDPM", "ReMiG"}

    try:
        if model_name in DIFFUSION_MODELS:
            # 扩散模型
            model_cls = _ddpm_dict[model_name]["model"]
            diffusion_cls = _ddpm_dict[model_name]["diffusion"]
            base_model = model_cls(args['model'])
            if model_name == "ReMiG":
                model = diffusion_cls(base_model, model_args=args['model'])
            else:
                model = diffusion_cls(
                    base_model,
                    model_args=args['model'],
                    schedule_opt=args.get('beta_schedule', {}).get('train', {})
                )
                model.set_new_noise_schedule(
                    args.get('beta_schedule', {}).get('train', {}),
                    device=device
                )
                model.set_loss(device)
        elif model_name == "Resshift":
            from utils.metrics import get_obj_from_str
            resshift_cfg = args.get('resshift', {})
            params = resshift_cfg.get("model", {}).get('params', {})
            model = get_obj_from_str(resshift_cfg['model']['target'])(**params)
        else:
            model = _model_dict[model_name](args['model'])
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to build model: {str(e)}",
            "recommendations": ["Check model configuration"]
        }

    model = model.to(device)
    model_params = sum(p.numel() for p in model.parameters())
    model_params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    # 确定输入尺寸
    if patch_size is not None:
        H, W = patch_size, patch_size
        h, w = patch_size // scale, patch_size // scale
    else:
        H, W = hr_shape[0], hr_shape[1]
        h, w = H // scale, W // scale

    # 构建 dummy 数据
    is_diffusion = model_name in DIFFUSION_MODELS
    is_resshift = model_name == "Resshift"

    # 构建优化器（需要用于 scaler.step）
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # 清空缓存并重置统计
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_id)

    result = {
        "gpu_name": gpu_name,
        "gpu_total_mb": round(gpu_total_mb, 1),
        "model_name": model_name,
        "model_params": model_params,
        "model_params_mb": round(model_params_mb, 2),
        "batch_size": batch_size,
        "input_shape_lr": [batch_size, h, w, n_channels],
        "input_shape_hr": [batch_size, H, W, n_channels],
        "use_amp": use_amp,
        "gradient_checkpointing": gradient_checkpointing,
        "patch_size": patch_size,
    }

    try:
        model.train()

        # 构造输入
        if is_diffusion:
            x = torch.randn(batch_size, n_channels, H, W, device=device)
            y = torch.randn(batch_size, n_channels, H, W, device=device)
            data = {'SR': x, 'HR': y}
        elif is_resshift:
            import torch.nn.functional as F
            x_lr = torch.randn(batch_size, n_channels, h, w, device=device)
            y = torch.randn(batch_size, n_channels, H, W, device=device)
            x = F.interpolate(x_lr, size=(H, W), mode='bicubic', align_corners=False)
        else:
            x = torch.randn(batch_size, h, w, n_channels, device=device)
            y = torch.randn(batch_size, H, W, n_channels, device=device)

        # Forward + Backward
        with torch.amp.autocast('cuda', enabled=use_amp):
            if is_diffusion:
                if gradient_checkpointing:
                    from torch.utils import checkpoint
                    def _forward(sr, hr):
                        return model({'SR': sr, 'HR': hr})
                    pix_loss = checkpoint.checkpoint(
                        _forward, data['SR'], data['HR'], use_reentrant=False
                    )
                else:
                    pix_loss = model(data)
                loss = pix_loss / max(batch_size * n_channels * H * W, 1)
            elif is_resshift:
                import functools
                from utils.metrics import get_obj_from_str as _get_obj
                resshift_cfg = args.get('resshift', {})
                params = resshift_cfg.get("diffusion", {}).get('params', {})
                base_diffusion = _get_obj(resshift_cfg['diffusion']['target'])(**params)
                tt = torch.randint(0, base_diffusion.num_timesteps, size=(batch_size,), device=device)
                model_kwargs = {'lq': x}
                losses, _, _ = base_diffusion.training_losses(
                    model, y, x, tt,
                    first_stage_model=None, model_kwargs=model_kwargs, noise=None,
                )
                loss = losses['mse']
            else:
                if gradient_checkpointing:
                    from torch.utils import checkpoint
                    y_pred = checkpoint.checkpoint(
                        model, x, use_reentrant=False
                    )
                else:
                    y_pred = model(x)
                loss = ((y_pred - y) ** 2).mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 记录峰值显存
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
        # 清理
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    return result


def _generate_recommendations(result):
    """根据测量结果生成优化建议"""
    recs = []
    pct = result["utilization_pct"]

    if pct > 90:
        recs.append(f"⚠️ 显存使用率 {pct}% 接近上限，训练中可能因波动 OOM")
        if not result["use_amp"]:
            recs.append("建议: 启用 use_amp=true 可减少约 40-50% 显存")
        if not result["gradient_checkpointing"]:
            recs.append("建议: 启用 gradient_checkpointing=true 可减少约 60% 激活显存")
        if result["patch_size"] is None:
            recs.append("建议: 设置 patch_size (如 64 或 128) 裁剪小区域训练")
        else:
            current_ps = result["patch_size"]
            recs.append(f"建议: 减小 patch_size (当前 {current_ps}，可尝试 {current_ps // 2})")
        recs.append(f"建议: 减小 batch_size (当前 {result['batch_size']})")
    elif pct > 75:
        recs.append(f"显存使用率 {pct}%，余量较小但可以训练")
        if not result["use_amp"]:
            recs.append("建议: 启用 use_amp=true 可进一步降低显存")
    elif pct > 50:
        recs.append(f"✅ 显存使用率 {pct}%，余量充足")
        recs.append(f"可尝试增大 batch_size 以提高训练速度")
    else:
        recs.append(f"✅ 显存使用率 {pct}%，余量很充足")
        recs.append(f"可显著增大 batch_size 或使用更大的模型")

    return recs


def _generate_oom_recommendations(result):
    """OOM 时的建议"""
    recs = [f"❌ 当前配置 OOM！GPU: {result['gpu_name']} ({result['gpu_total_mb']:.0f} MB)"]

    applied = []
    if result["use_amp"]:
        applied.append("AMP")
    if result["gradient_checkpointing"]:
        applied.append("gradient checkpointing")
    if result["patch_size"]:
        applied.append(f"patch_size={result['patch_size']}")

    if applied:
        recs.append(f"已启用: {', '.join(applied)}")

    if not result["use_amp"]:
        recs.append("1. 启用 use_amp=true（减少约 40-50% 显存）")
    if not result["gradient_checkpointing"]:
        recs.append("2. 启用 gradient_checkpointing=true（减少约 60% 激活显存）")

    bs = result["batch_size"]
    if bs > 1:
        recs.append(f"3. 减小 batch_size: {bs} → {max(bs // 2, 1)}")

    if result["patch_size"] is None:
        recs.append("4. 设置 patch_size=64 或 128 裁剪小区域训练")
    else:
        ps = result["patch_size"]
        if ps > 32:
            recs.append(f"4. 减小 patch_size: {ps} → {ps // 2}")

    return recs


def main():
    parser = argparse.ArgumentParser(description='Estimate GPU memory for training')
    parser.add_argument('--config', type=str, required=True, help='Training config YAML path')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    result = estimate_memory(args.config, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
