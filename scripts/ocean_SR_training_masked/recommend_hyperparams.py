"""
recommend_hyperparams.py

@description 超参数推荐脚本。根据实际 GPU 显存探针测量和数据集统计，
    推荐 batch_size、epochs、lr，并附带数据频谱分析（供用户参考）。

@author Leizheng
@date 2026-02-25
@version 1.0.0

@changelog
  - 2026-02-25 Leizheng: v1.0.0 初始版本
    - 通过 batch=1 探针运行 estimate_memory.py 精确测量单样本显存
    - 从测量值推算最大 batch_size（取 2 的幂次，含 70% 安全系数）
    - 基于数据集大小 + 模型类型推荐 epochs
    - 基于 batch_size 平方根缩放推荐 lr
    - 附带简单 2D 频谱分析（k90 指标），供用户判断是否需要调整 FNO modes 等

Usage:
    python recommend_hyperparams.py \\
        --dataset_root /path/to/dataset \\
        --model_name ResShift \\
        --scale 4 \\
        --dyn_vars temp,salt \\
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

# 扩散模型集合（收敛慢，需要更多 epoch）
DIFFUSION_MODELS = {'DDPM', 'SR3', 'MG-DDPM', 'Resshift', 'ResShift', 'ReMiG'}

# 扩散模型推理显存大（2000步采样），eval_batch_size 上限
DIFFUSION_EVAL_BATCH_MAX = 4

# 各模型在 reference batch=4 下的基准学习率（来自各自论文或常用配置）
BASE_LR: dict[str, float] = {
    'FNO2d':               1e-3,
    'UNet2d':              1e-3,
    'M2NO2d':              1e-3,
    'Galerkin_Transformer': 1e-3,
    'MWT2d':               1e-3,
    'SRNO':                1e-3,
    'Swin_Transformer':    2e-4,
    'EDSR':                1e-4,
    'HiNOTE':              1e-3,
    'SwinIR':              2e-4,
    'DDPM':                1e-4,
    'SR3':                 1e-4,
    'MG-DDPM':             1e-4,
    'Resshift':            5e-5,
    'ResShift':            5e-5,
    'ReMiG':               1e-4,
}
BASE_BATCH = 4  # 以上 BASE_LR 对应的参考 batch_size


# ---------------------------------------------------------------------------
# 数据集扫描
# ---------------------------------------------------------------------------

def scan_dataset(dataset_root: str, dyn_vars: list[str]) -> dict:
    """统计训练集样本数与 HR 分辨率。"""
    first_var = dyn_vars[0]
    result: dict = {'n_train': 0, 'n_valid': 0, 'n_test': 0, 'hr_shape': None}

    for split in ('train', 'valid', 'test'):
        hr_dir = os.path.join(dataset_root, split, 'hr', first_var)
        if os.path.isdir(hr_dir):
            files = sorted(glob.glob(os.path.join(hr_dir, '*.npy')))
            result[f'n_{split}'] = len(files)
            if result['hr_shape'] is None and files:
                arr = np.load(files[0])
                result['hr_shape'] = list(arr.shape[:2])  # [H, W]

    return result


# ---------------------------------------------------------------------------
# GPU 探针测量
# ---------------------------------------------------------------------------

def probe_memory(
    model_name: str,
    dataset_root: str,
    dyn_vars: list[str],
    scale: int,
    device: int,
) -> tuple[dict | None, dict]:
    """
    生成 batch_size=1 的探针配置，调用 estimate_memory.py 测量单样本显存。

    Returns:
        (mem_data, gpu_info_basic)
        mem_data: estimate_memory.py 的输出 dict，失败时为 None
        gpu_info_basic: {'name': str, 'total_mb': float}
    """
    # 先获取 GPU 基本信息（不依赖探针是否成功）
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

    # 生成探针配置
    probe_dir = tempfile.mkdtemp(prefix='ocean_sr_probe_')
    probe_config = os.path.join(probe_dir, 'config.yaml')

    probe_params = {
        'model_name': model_name,
        'dataset_root': dataset_root,
        'dyn_vars': dyn_vars,
        'scale': scale,
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
# 频谱分析
# ---------------------------------------------------------------------------

def analyze_spectrum(dataset_root: str, dyn_vars: list[str], n_samples: int = 10) -> dict | None:
    """
    对训练集 HR 数据做简单的 2D 频谱分析，计算 k90（包含 90% 能量的截止频率）。

    不用于自动调参，仅作为参考信息展示给用户（如 FNO modes 的选择参考）。
    """
    first_var = dyn_vars[0]
    hr_dir = os.path.join(dataset_root, 'train', 'hr', first_var)
    if not os.path.isdir(hr_dir):
        return None

    files = sorted(glob.glob(os.path.join(hr_dir, '*.npy')))
    if not files:
        return None
    files = files[:n_samples]

    k90_list: list[float] = []
    max_k = 0

    for f in files:
        try:
            data = np.load(f)
            if data.ndim == 3:
                data = data[..., 0]
            if data.ndim != 2:
                continue

            H, W = data.shape
            max_k = min(H // 2, W // 2)

            # NaN → 均值填充（避免 FFT 异常）
            nan_mean = float(np.nanmean(data))
            data = np.where(np.isnan(data), nan_mean, data)

            # 2D FFT → 功率谱 → 移到中心
            F = np.fft.fft2(data - data.mean())
            power = np.fft.fftshift(np.abs(F) ** 2)

            # 径向平均
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
# 推荐计算
# ---------------------------------------------------------------------------

def recommend_batch(mem_data: dict | None, total_mb: float) -> tuple[int | None, str]:
    """根据探针测量结果推算 batch_size。"""
    if mem_data is None:
        return None, '探针测量失败，无法自动推荐 batch_size'

    if mem_data.get('status') == 'oom':
        return 1, 'batch=1 时已 OOM，建议减小 patch_size 或模型规模'

    if mem_data.get('status') == 'error':
        return None, f"探针测量出错：{mem_data.get('error', '未知错误')}"

    peak_mb = float(mem_data.get('peak_memory_mb') or 0)
    model_params_mb = float(mem_data.get('model_params_mb') or 0)

    if peak_mb <= 0 or total_mb <= 0:
        return None, '显存数据缺失，无法推荐 batch_size'

    # 激活显存（不含模型参数，随 batch 线性扩展）
    activation_per_sample = max(peak_mb - model_params_mb, peak_mb * 0.3)

    usable_mb = total_mb * 0.70  # 保留 30% 安全余量
    max_batch_float = (usable_mb - model_params_mb) / activation_per_sample
    max_batch = max(1, int(max_batch_float))

    # 取不超过 max_batch 的最大 2^n（上限 32）
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
    """根据训练样本数和模型类型推荐 epoch 数。"""
    is_diffusion = model_name in DIFFUSION_MODELS

    if n_train < 500:
        base = 1200 if is_diffusion else 800
        bucket = f'小数据集（{n_train} 样本 < 500）'
    elif n_train < 2000:
        base = 800 if is_diffusion else 500
        bucket = f'中小数据集（{n_train} 样本）'
    elif n_train < 10000:
        base = 500 if is_diffusion else 300
        bucket = f'中等数据集（{n_train} 样本）'
    else:
        base = 200 if is_diffusion else 150
        bucket = f'大数据集（{n_train} 样本）'

    diffusion_note = '，扩散模型收敛慢基准较高' if is_diffusion else ''
    reasoning = f'{bucket}{diffusion_note}，推荐 {base} epoch'
    return base, reasoning


def recommend_lr(batch_size: int, model_name: str) -> tuple[float, str]:
    """从 reference batch=4 对应的基准 LR，按平方根规则缩放到实际 batch_size。"""
    base_lr = BASE_LR.get(model_name, 1e-3)

    if batch_size != BASE_BATCH and batch_size > 0:
        lr = base_lr * math.sqrt(batch_size / BASE_BATCH)
        # 保留 2 位有效数字
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
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Ocean SR 超参数推荐')
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--dyn_vars', required=True,
                        help='逗号分隔的变量名，如 temp,salt')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    dyn_vars = [v.strip() for v in args.dyn_vars.split(',') if v.strip()]
    output: dict = {'status': 'success'}

    # ------------------------------------------------------------------
    # 1. 扫描数据集
    # ------------------------------------------------------------------
    try:
        ds = scan_dataset(args.dataset_root, dyn_vars)
        output['dataset_info'] = {
            'n_train': ds['n_train'],
            'n_valid': ds['n_valid'],
            'n_test':  ds['n_test'],
            'hr_shape': ds['hr_shape'],
            'n_vars': len(dyn_vars),
        }
    except Exception as e:
        output['status'] = 'error'
        output['error'] = f'数据集扫描失败: {e}'
        print(f"__recommend__{json.dumps(output, ensure_ascii=False)}__recommend__",
              flush=True)
        return

    # ------------------------------------------------------------------
    # 2. GPU 探针测量
    # ------------------------------------------------------------------
    mem_data, gpu_info_basic = probe_memory(
        args.model_name, args.dataset_root, dyn_vars,
        args.scale, args.device,
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
    # 3. 频谱分析（仅供参考，不影响推荐值）
    # ------------------------------------------------------------------
    try:
        spectral = analyze_spectrum(args.dataset_root, dyn_vars)
        if spectral:
            output['spectral_analysis'] = spectral
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4. 计算推荐值
    # ------------------------------------------------------------------
    recommendations: dict = {}
    reasoning: dict = {}

    # batch_size
    rec_batch, batch_reason = recommend_batch(mem_data, total_mb)
    if rec_batch is not None:
        recommendations['batch_size'] = rec_batch
        # 扩散模型推理显存大，eval_batch 上限 4
        if args.model_name in DIFFUSION_MODELS:
            recommendations['eval_batch_size'] = min(rec_batch, DIFFUSION_EVAL_BATCH_MAX)
        else:
            recommendations['eval_batch_size'] = rec_batch
        reasoning['batch_size'] = batch_reason

    # epochs
    n_train = ds['n_train']
    rec_epochs, epoch_reason = recommend_epochs(n_train, args.model_name)
    recommendations['epochs'] = rec_epochs
    reasoning['epochs'] = epoch_reason

    # lr（batch_size 已知则缩放，否则用 BASE_BATCH 基准）
    effective_batch = rec_batch if rec_batch is not None else BASE_BATCH
    rec_lr, lr_reason = recommend_lr(effective_batch, args.model_name)
    recommendations['lr'] = rec_lr
    reasoning['lr'] = lr_reason

    # gradient_checkpointing（显存紧张时建议开启）
    if total_mb > 0 and total_mb < 24 * 1024:
        recommendations['gradient_checkpointing'] = True
        reasoning['gradient_checkpointing'] = (
            f'GPU 显存 {total_mb / 1024:.0f}GB < 24GB，建议开启以防 OOM'
        )

    # ------------------------------------------------------------------
    # 5. 模型特定提示（不自动修改，仅文字说明）
    # ------------------------------------------------------------------
    model_notes: dict = {}
    spectral = output.get('spectral_analysis')
    if args.model_name == 'FNO2d' and spectral:
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

    if model_notes:
        output['model_notes'] = model_notes

    output['recommendations'] = recommendations
    output['reasoning'] = reasoning

    print(f"__recommend__{json.dumps(output, ensure_ascii=False)}__recommend__",
          flush=True)


if __name__ == '__main__':
    main()
