"""
@file recommend_hyperparams.py
@description Hyperparameter recommendation for ocean forecast training.
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
    - Probe with batch=1 via estimate_memory.py to measure per-sample GPU memory
    - Derive max batch_size (power-of-2, 70% safety margin)
    - Recommend epochs based on dataset size and model type
    - Recommend lr via sqrt scaling from base batch=4 reference
    - 2D spectrum analysis (k90 metric) for user reference (e.g. FNO modes)
    - No scale parameter (forecast = same resolution in/out)
    - Uses in_t/out_t/stride for time-series config generation

Usage:
    python recommend_hyperparams.py \\
        --dataset_root /path/to/dataset \\
        --model_name FNO2d \\
        --dyn_vars uo,vo \\
        --in_t 7 \\
        --out_t 1 \\
        --device 0

Output: __recommend__{JSON}__recommend__
"""

import os
import sys
import json
import math
import glob
import shutil
import tempfile
import subprocess
import argparse

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base learning rates at reference batch=4 for each forecast model
# (from respective papers or common configurations)
BASE_LR: dict[str, float] = {
    'FNO2d':                1e-3,
    'UNet2d':               1e-3,
    'SwinTransformerV2':    2e-4,
    'SwinMLP':              2e-4,
    'Transformer':          3e-4,
    'M2NO2d':               1e-3,
    'GalerkinTransformer':  1e-3,
    'Transolver':           1e-3,
    'GNOT':                 1e-3,
    'ONO':                  1e-3,
    'LSM':                  1e-3,
    'LNO':                  1e-3,
    'MLP':                  1e-3,
    'UNet1d':               1e-3,
    'FNO1d':                1e-3,
    'UNet3d':               1e-3,
    'FNO3d':                1e-3,
}
BASE_BATCH = 4  # Reference batch_size that the above BASE_LR values correspond to


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

def scan_dataset(dataset_root: str, dyn_vars: list[str]) -> dict:
    """
    Scan dataset to count training/validation/test samples and detect spatial shape.

    Forecast directory structure: {split}/{var_name}/{date}.npy
    """
    first_var = dyn_vars[0]
    result: dict = {'n_train': 0, 'n_valid': 0, 'n_test': 0, 'spatial_shape': None}

    for split in ('train', 'valid', 'test'):
        var_dir = os.path.join(dataset_root, split, first_var)
        if os.path.isdir(var_dir):
            files = sorted(glob.glob(os.path.join(var_dir, '*.npy')))
            result[f'n_{split}'] = len(files)
            if result['spatial_shape'] is None and files:
                arr = np.load(files[0])
                result['spatial_shape'] = list(arr.shape[:2])  # [H, W]

    return result


def load_var_names_json(dataset_root: str) -> dict | None:
    """Load var_names.json from dataset_root if it exists."""
    var_names_path = os.path.join(dataset_root, 'var_names.json')
    if os.path.isfile(var_names_path):
        with open(var_names_path, 'r') as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# GPU probe measurement
# ---------------------------------------------------------------------------

def probe_memory(
    model_name: str,
    dataset_root: str,
    dyn_vars: list[str],
    in_t: int,
    out_t: int,
    device: int,
) -> tuple[dict | None, dict]:
    """
    Generate a probe config with batch_size=1 and invoke estimate_memory.py
    to measure per-sample GPU memory.

    Returns:
        (mem_data, gpu_info_basic)
        mem_data: estimate_memory.py output dict, None on failure
        gpu_info_basic: {'name': str, 'total_mb': float}
    """
    # Get basic GPU info (independent of probe success)
    gpu_name = 'Unknown GPU'
    total_mb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            gpu_name = props.name
            total_mb = props.total_memory / 1024 ** 2
    except Exception:
        pass

    gpu_info_basic = {'name': gpu_name, 'total_mb': total_mb}

    # Generate probe config
    probe_dir = tempfile.mkdtemp(prefix='ocean_forecast_probe_')
    probe_config = os.path.join(probe_dir, 'config.yaml')

    probe_params = {
        'model_name': model_name,
        'dataset_root': dataset_root,
        'dyn_vars': dyn_vars,
        'in_t': in_t,
        'out_t': out_t,
        'stride': 1,
        'batch_size': 1,
        'eval_batch_size': 1,
        'log_dir': probe_dir,
        'epochs': 1,
    }

    try:
        gen_script = os.path.join(SCRIPT_DIR, 'generate_config.py')
        gen_result = subprocess.run(
            [sys.executable, gen_script,
             '--params', json.dumps(probe_params),
             '--output', probe_config],
            capture_output=True, text=True,
            cwd=SCRIPT_DIR, timeout=60,
        )
        if gen_result.returncode != 0:
            return None, gpu_info_basic

        mem_script = os.path.join(SCRIPT_DIR, 'estimate_memory.py')
        if not os.path.isfile(mem_script):
            # Fallback: try the SR version's estimate_memory.py
            sr_mem_script = os.path.join(
                os.path.dirname(SCRIPT_DIR),
                'ocean_SR_training_masked', 'estimate_memory.py',
            )
            if os.path.isfile(sr_mem_script):
                mem_script = sr_mem_script
            else:
                return None, gpu_info_basic

        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(device)}
        mem_result = subprocess.run(
            [sys.executable, mem_script,
             '--config', probe_config,
             '--device', '0'],
            capture_output=True, text=True,
            cwd=SCRIPT_DIR, timeout=180, env=env,
        )

        try:
            mem_data = json.loads(mem_result.stdout.strip())
        except Exception:
            return None, gpu_info_basic

        return mem_data, gpu_info_basic

    except Exception:
        return None, gpu_info_basic
    finally:
        try:
            shutil.rmtree(probe_dir)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Spectrum analysis
# ---------------------------------------------------------------------------

def analyze_spectrum(dataset_root: str, dyn_vars: list[str], n_samples: int = 10) -> dict | None:
    """
    Simple 2D spectrum analysis on training data to compute k90 (cutoff frequency
    containing 90% of spectral energy).

    Not used for automatic tuning -- only displayed as reference for the user
    (e.g. for choosing FNO modes).

    Forecast directory structure: train/{var_name}/*.npy
    """
    first_var = dyn_vars[0]
    var_dir = os.path.join(dataset_root, 'train', first_var)
    if not os.path.isdir(var_dir):
        return None

    files = sorted(glob.glob(os.path.join(var_dir, '*.npy')))
    if not files:
        return None
    files = files[:n_samples]

    k90_list: list[float] = []
    max_k = 0

    for f in files:
        try:
            data = np.load(f)
            # Each file is a single timestep [H, W] or [H, W, C]
            if data.ndim == 3:
                data = data[..., 0]
            if data.ndim != 2:
                continue

            H, W = data.shape
            max_k = min(H // 2, W // 2)

            # NaN -> mean fill (avoid FFT anomalies)
            nan_mean = float(np.nanmean(data))
            data = np.where(np.isnan(data), nan_mean, data)

            # 2D FFT -> power spectrum -> shift to center
            F = np.fft.fft2(data - data.mean())
            power = np.fft.fftshift(np.abs(F) ** 2)

            # Radial average
            cy, cx = H // 2, W // 2
            Y, X = np.ogrid[:H, :W]
            R = np.round(np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)).astype(int)

            radial = np.array([
                power[R == r].mean() if (R == r).any() else 0.0
                for r in range(max_k)
            ])

            total = radial.sum()
            if total <= 0:
                continue

            cumulative = np.cumsum(radial) / total
            k90 = int(np.searchsorted(cumulative, 0.90))
            k90_list.append(float(k90))

        except Exception:
            continue

    if not k90_list or max_k == 0:
        return None

    k90_mean = float(np.mean(k90_list))
    ratio = k90_mean / max_k

    if ratio < 0.20:
        freq_type = 'low'
        freq_desc = '低频为主（大尺度环流特征）'
    elif ratio < 0.45:
        freq_type = 'mixed'
        freq_desc = '中频为主（含中尺度涡旋）'
    else:
        freq_type = 'high'
        freq_desc = '高频为主（含细尺度结构 / 锋面）'

    return {
        'k90_mean': round(k90_mean, 1),
        'max_k': max_k,
        'freq_type': freq_type,
        'freq_desc': freq_desc,
    }


# ---------------------------------------------------------------------------
# Recommendation computations
# ---------------------------------------------------------------------------

def recommend_batch(mem_data: dict | None, total_mb: float) -> tuple[int | None, str]:
    """Derive recommended batch_size from probe measurement results."""
    if mem_data is None:
        return None, '探针测量失败，无法自动推荐 batch_size'

    if mem_data.get('status') == 'oom':
        return 1, 'batch=1 时已 OOM，建议减小模型规模或输入时间步长'

    if mem_data.get('status') == 'error':
        return None, f"探针测量出错：{mem_data.get('error', mem_data.get('message', '未知错误'))}"

    peak_mb = float(mem_data.get('peak_memory_mb') or 0)
    model_params_mb = float(mem_data.get('model_params_mb') or 0)

    if peak_mb <= 0 or total_mb <= 0:
        return None, '显存数据缺失，无法推荐 batch_size'

    # Activation memory (excl. model params, scales linearly with batch)
    activation_per_sample = max(peak_mb - model_params_mb, peak_mb * 0.3)

    usable_mb = total_mb * 0.70  # Reserve 30% safety margin
    max_batch_float = (usable_mb - model_params_mb) / activation_per_sample
    max_batch = max(1, int(max_batch_float))

    # Largest power-of-2 not exceeding max_batch (capped at 32)
    if max_batch >= 2:
        recommended = 2 ** int(math.log2(max_batch))
    else:
        recommended = 1
    recommended = min(recommended, 32)

    reasoning = (
        f"单样本激活显存 ≈ {activation_per_sample:.0f}MB，"
        f"模型参数 ≈ {model_params_mb:.0f}MB，"
        f"GPU {total_mb / 1024:.0f}GB 保留 30% 余量后可用 {usable_mb / 1024:.1f}GB，"
        f"推荐 batch_size={recommended}"
    )
    return recommended, reasoning


def recommend_epochs(n_train: int, model_name: str) -> tuple[int, str]:
    """Recommend epoch count based on training set size and model type."""
    if n_train < 500:
        base = 800
        bucket = f'小数据集（{n_train} 样本 < 500）'
    elif n_train < 2000:
        base = 500
        bucket = f'中小数据集（{n_train} 样本）'
    elif n_train < 10000:
        base = 300
        bucket = f'中等数据集（{n_train} 样本）'
    else:
        base = 150
        bucket = f'大数据集（{n_train} 样本）'

    reasoning = f'{bucket}，推荐 {base} epoch'
    return base, reasoning


def recommend_lr(batch_size: int, model_name: str) -> tuple[float, str]:
    """Scale base LR (at reference batch=4) to actual batch_size via sqrt rule."""
    base_lr = BASE_LR.get(model_name, 1e-3)

    if batch_size != BASE_BATCH and batch_size > 0:
        lr = base_lr * math.sqrt(batch_size / BASE_BATCH)
        # Round to 2 significant figures
        if lr > 0:
            magnitude = 10 ** math.floor(math.log10(lr))
            lr = round(lr / magnitude, 2) * magnitude
    else:
        lr = base_lr

    reasoning = (
        f'{model_name} 基准 lr={base_lr:.0e}（batch={BASE_BATCH}），'
        f'batch={batch_size} 时平方根缩放 → lr={lr:.2e}'
    )
    return lr, reasoning


# ---------------------------------------------------------------------------
# Memory estimation helpers (heuristic fallback when probe fails)
# ---------------------------------------------------------------------------

def estimate_memory_heuristic(
    model_name: str,
    spatial_shape: list[int],
    in_t: int,
    out_t: int,
    n_vars: int,
) -> dict:
    """
    Rough heuristic memory estimate when the probe cannot run.
    Input tensor size: H * W * in_t * n_vars (forecast: same resolution in/out).
    """
    H, W = spatial_shape[0], spatial_shape[1]
    input_elements = H * W * in_t * n_vars
    output_elements = H * W * out_t * n_vars

    # Rough estimate: 4 bytes/element * (input + output + activations ~4x input)
    est_mb = (input_elements + output_elements + 4 * input_elements) * 4 / (1024 ** 2)

    return {
        'input_elements': input_elements,
        'output_elements': output_elements,
        'estimated_activation_mb': round(est_mb, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Ocean forecast hyperparameter recommendation')
    parser.add_argument('--dataset_root', required=True,
                        help='Root directory of preprocessed forecast dataset')
    parser.add_argument('--model_name', required=True,
                        help='Model name, e.g. FNO2d, UNet2d')
    parser.add_argument('--dyn_vars', required=True,
                        help='Comma-separated dynamic variable names, e.g. uo,vo')
    parser.add_argument('--in_t', type=int, default=7,
                        help='Input timesteps (default: 7)')
    parser.add_argument('--out_t', type=int, default=1,
                        help='Output timesteps (default: 1)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (default: 0)')
    args = parser.parse_args()

    dyn_vars = [v.strip() for v in args.dyn_vars.split(',') if v.strip()]
    output: dict = {'status': 'success'}

    # ------------------------------------------------------------------
    # 0. Try loading var_names.json for supplementary info
    # ------------------------------------------------------------------
    var_names_info = load_var_names_json(args.dataset_root)
    spatial_shape_from_json = None
    if var_names_info:
        spatial_shape_from_json = var_names_info.get('spatial_shape', None)
        # If dyn_vars not provided via CLI, try var_names.json
        if not dyn_vars:
            dyn_vars = var_names_info.get('dynamic', var_names_info.get('dyn_vars', []))

    # ------------------------------------------------------------------
    # 1. Scan dataset
    # ------------------------------------------------------------------
    try:
        ds = scan_dataset(args.dataset_root, dyn_vars)
        # Prefer spatial_shape from var_names.json if available
        spatial_shape = spatial_shape_from_json or ds['spatial_shape']
        output['dataset_info'] = {
            'n_train': ds['n_train'],
            'n_valid': ds['n_valid'],
            'n_test':  ds['n_test'],
            'spatial_shape': spatial_shape,
            'n_vars': len(dyn_vars),
            'in_t': args.in_t,
            'out_t': args.out_t,
        }
    except Exception as e:
        output['status'] = 'error'
        output['error'] = f'数据集扫描失败: {e}'
        print(f"__recommend__{json.dumps(output, ensure_ascii=False)}__recommend__",
              flush=True)
        return

    # ------------------------------------------------------------------
    # 2. GPU probe measurement
    # ------------------------------------------------------------------
    mem_data, gpu_info_basic = probe_memory(
        args.model_name, args.dataset_root, dyn_vars,
        args.in_t, args.out_t, args.device,
    )
    total_mb = gpu_info_basic['total_mb']

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(args.device)
            output['gpu_info'] = {
                'name':     props.name,
                'total_mb': round(total_mb),
                'total_gb': round(total_mb / 1024, 1),
            }
        else:
            output['gpu_info'] = {'name': 'CPU（无 CUDA）', 'total_mb': 0}
    except Exception:
        output['gpu_info'] = gpu_info_basic

    # ------------------------------------------------------------------
    # 3. Spectrum analysis (reference only, does not affect recommendations)
    # ------------------------------------------------------------------
    try:
        spectral = analyze_spectrum(args.dataset_root, dyn_vars)
        if spectral:
            output['spectral_analysis'] = spectral
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4. Heuristic memory info (always included for reference)
    # ------------------------------------------------------------------
    if spatial_shape:
        heuristic = estimate_memory_heuristic(
            args.model_name, spatial_shape,
            args.in_t, args.out_t, len(dyn_vars),
        )
        output['memory_heuristic'] = heuristic

    # ------------------------------------------------------------------
    # 5. Compute recommendations
    # ------------------------------------------------------------------
    recommendations: dict = {}
    reasoning: dict = {}

    # batch_size
    rec_batch, batch_reason = recommend_batch(mem_data, total_mb)
    if rec_batch is not None:
        recommendations['batch_size'] = rec_batch
        recommendations['eval_batch_size'] = rec_batch
        reasoning['batch_size'] = batch_reason

    # epochs
    n_train = ds['n_train']
    rec_epochs, epoch_reason = recommend_epochs(n_train, args.model_name)
    recommendations['epochs'] = rec_epochs
    reasoning['epochs'] = epoch_reason

    # lr (scale from base if batch_size is known, else use BASE_BATCH reference)
    effective_batch = rec_batch if rec_batch is not None else BASE_BATCH
    rec_lr, lr_reason = recommend_lr(effective_batch, args.model_name)
    recommendations['lr'] = rec_lr
    reasoning['lr'] = lr_reason

    # gradient_checkpointing (suggest when GPU memory is tight)
    if total_mb > 0 and total_mb < 24 * 1024:
        recommendations['gradient_checkpointing'] = True
        reasoning['gradient_checkpointing'] = (
            f'GPU 显存 {total_mb / 1024:.0f}GB < 24GB，建议开启以防 OOM'
        )

    # in_t / out_t (echo back for clarity)
    recommendations['in_t'] = args.in_t
    recommendations['out_t'] = args.out_t

    # ------------------------------------------------------------------
    # 6. Model-specific notes (informational only, no auto-adjustment)
    # ------------------------------------------------------------------
    model_notes: dict = {}
    spectral = output.get('spectral_analysis')
    if args.model_name in ('FNO2d', 'FNO1d', 'FNO3d') and spectral:
        k90 = spectral['k90_mean']
        if k90 > 15:
            model_notes['fno_modes'] = (
                f'数据 k90≈{k90:.0f}，当前默认 modes=15 可能丢失部分高频细节，'
                f'可考虑适当增大（如 20-{int(k90) + 5}）'
            )
        else:
            model_notes['fno_modes'] = (
                f'数据 k90≈{k90:.0f}，当前默认 modes=15 已足够覆盖主要频率'
            )

    if args.model_name in ('SwinTransformerV2', 'SwinMLP') and spatial_shape:
        H, W = spatial_shape[0], spatial_shape[1]
        if H % 8 != 0 or W % 8 != 0:
            model_notes['spatial_shape'] = (
                f'Swin 系列要求空间维度能被 window_size 整除，'
                f'当前 H={H}, W={W} 可能需要 padding 或裁剪'
            )

    if model_notes:
        output['model_notes'] = model_notes

    output['recommendations'] = recommendations
    output['reasoning'] = reasoning

    print(f"__recommend__{json.dumps(output, ensure_ascii=False)}__recommend__",
          flush=True)


if __name__ == '__main__':
    main()
