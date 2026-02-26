#!/usr/bin/env python3
"""
@file generate_training_plots.py
@description Generate training visualization plots for ocean forecast models.
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print(json.dumps({
        "status": "error",
        "errors": ["matplotlib is required: pip install matplotlib"]
    }))
    sys.exit(1)

# ---------------------------------------------------------------------------
# Font configuration: Chinese + negative-sign support
# ---------------------------------------------------------------------------
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# Modern style
# ---------------------------------------------------------------------------
_STYLE_APPLIED = False
for _s in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid']:
    if _s in plt.style.available:
        plt.style.use(_s)
        _STYLE_APPLIED = True
        break
if not _STYLE_APPLIED:
    # Fallback: manually set a clean look
    matplotlib.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': '#f8f8f8',
        'figure.facecolor': 'white',
    })

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLOR_TRAIN_LOSS = '#2196F3'   # blue
COLOR_VALID_LOSS = '#F44336'   # red
COLOR_RMSE = '#4CAF50'         # green
COLOR_MAE = '#FF9800'          # orange
COLOR_LR = '#9C27B0'           # purple
COLOR_BAR_BASE = '#42A5F5'     # light blue for bars

# =========================================================================
# 1. Log parsing
# =========================================================================

_EVENT_RE = re.compile(r'__event__(\{.*?\})__event__')


def parse_log_file(log_dir: str) -> List[Dict[str, Any]]:
    """
    Scan all ``*.log`` and ``*.txt`` files under *log_dir* for lines that
    contain ``__event__{json}__event__`` markers and return the parsed
    JSON objects as a list, preserving file/line order.
    """
    events: List[Dict[str, Any]] = []

    log_files: List[str] = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith('.log') or fname.endswith('.txt'):
            log_files.append(os.path.join(log_dir, fname))

    if not log_files:
        # Fallback: try any file that is not a directory or image
        for fname in sorted(os.listdir(log_dir)):
            fpath = os.path.join(log_dir, fname)
            if os.path.isfile(fpath) and not fname.endswith(('.png', '.jpg', '.npy', '.pth')):
                log_files.append(fpath)

    for fpath in log_files:
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    for m in _EVENT_RE.finditer(line):
                        try:
                            obj = json.loads(m.group(1))
                            events.append(obj)
                        except json.JSONDecodeError:
                            pass
        except Exception:
            continue

    return events


def extract_training_data(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Walk the event list once and bucket data into handy structures:

    Returns dict with keys:
      - epochs_train: list of (epoch, train_loss, lr)
      - epochs_valid: list of (epoch, valid_loss, metrics_dict, per_var_metrics_or_None)
      - final_test: dict or None
      - training_start: dict or None
      - training_end: dict or None
    """
    epochs_train: List[Dict[str, Any]] = []
    epochs_valid: List[Dict[str, Any]] = []
    final_test: Optional[Dict[str, Any]] = None
    training_start: Optional[Dict[str, Any]] = None
    training_end: Optional[Dict[str, Any]] = None

    for ev in events:
        etype = ev.get('type', '')
        if etype == 'epoch_train':
            epochs_train.append(ev)
        elif etype == 'epoch_valid':
            epochs_valid.append(ev)
        elif etype == 'final_test':
            final_test = ev
        elif etype == 'training_start':
            training_start = ev
        elif etype == 'training_end':
            training_end = ev

    return {
        'epochs_train': epochs_train,
        'epochs_valid': epochs_valid,
        'final_test': final_test,
        'training_start': training_start,
        'training_end': training_end,
    }


# =========================================================================
# 2. Individual plot functions
# =========================================================================

def plot_loss_curve(
    train_data: List[Dict[str, Any]],
    valid_data: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """
    Plot 1: Train / validation loss over epochs.
    Saves to ``out_path`` (e.g. ``plots/loss_curve.png``).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Train loss
    if train_data:
        t_epochs = [d['epoch'] for d in train_data]
        t_loss = [d['train_loss'] for d in train_data]
        ax.plot(t_epochs, t_loss, color=COLOR_TRAIN_LOSS, linewidth=1.5,
                label='Train Loss', alpha=0.85)

    # Valid loss
    if valid_data:
        v_epochs = [d['epoch'] for d in valid_data]
        v_loss = [d['valid_loss'] for d in valid_data]
        ax.plot(v_epochs, v_loss, color=COLOR_VALID_LOSS, linewidth=1.5,
                marker='o', markersize=3, label='Valid Loss', alpha=0.85)

        # Mark best valid loss
        if v_loss:
            best_idx = int(np.argmin(v_loss))
            ax.annotate(
                f'Best: {v_loss[best_idx]:.4e}\n(epoch {v_epochs[best_idx]})',
                xy=(v_epochs[best_idx], v_loss[best_idx]),
                xytext=(20, 20), textcoords='offset points',
                fontsize=8, color=COLOR_VALID_LOSS,
                arrowprops=dict(arrowstyle='->', color=COLOR_VALID_LOSS, lw=1.0),
            )

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training / Validation Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_curve(
    valid_data: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """
    Plot 2: RMSE and MAE metrics over epochs (dual y-axis).
    Saves to ``out_path`` (e.g. ``plots/metrics_curve.png``).
    """
    if not valid_data:
        return

    epochs: List[int] = []
    rmse_vals: List[float] = []
    mae_vals: List[float] = []

    for d in valid_data:
        metrics = d.get('metrics', {})
        ep = d['epoch']
        r = metrics.get('rmse')
        m = metrics.get('mae')
        if r is not None or m is not None:
            epochs.append(ep)
            rmse_vals.append(float(r) if r is not None else float('nan'))
            mae_vals.append(float(m) if m is not None else float('nan'))

    if not epochs:
        return

    has_rmse = any(math.isfinite(v) for v in rmse_vals)
    has_mae = any(math.isfinite(v) for v in mae_vals)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    if has_rmse:
        ax1.plot(epochs, rmse_vals, color=COLOR_RMSE, linewidth=1.5,
                 marker='s', markersize=3, label='RMSE', alpha=0.85)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('RMSE', fontsize=11, color=COLOR_RMSE)
    ax1.tick_params(axis='y', labelcolor=COLOR_RMSE, labelsize=9)
    ax1.tick_params(axis='x', labelsize=9)

    if has_mae:
        ax2 = ax1.twinx()
        ax2.plot(epochs, mae_vals, color=COLOR_MAE, linewidth=1.5,
                 marker='^', markersize=3, label='MAE', alpha=0.85)
        ax2.set_ylabel('MAE', fontsize=11, color=COLOR_MAE)
        ax2.tick_params(axis='y', labelcolor=COLOR_MAE, labelsize=9)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_mae:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    else:
        ax1.legend(fontsize=9, loc='upper right')

    ax1.set_title('Validation Metrics (RMSE / MAE)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_lr_curve(
    train_data: List[Dict[str, Any]],
    out_path: str,
) -> None:
    """
    Plot 3: Learning rate schedule over epochs.
    Saves to ``out_path`` (e.g. ``plots/lr_curve.png``).
    """
    if not train_data:
        return

    epochs = [d['epoch'] for d in train_data]
    lrs = [d.get('lr', 0.0) for d in train_data]

    if not any(lr > 0 for lr in lrs):
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(epochs, lrs, color=COLOR_LR, linewidth=1.5, alpha=0.85)
    ax.fill_between(epochs, 0, lrs, color=COLOR_LR, alpha=0.10)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_var_metrics(
    valid_data: List[Dict[str, Any]],
    final_test: Optional[Dict[str, Any]],
    out_path: str,
) -> None:
    """
    Plot 4: Per-variable RMSE bar chart.

    Prefers ``final_test.per_var_metrics`` if available; otherwise falls back
    to the last ``epoch_valid`` event that carries ``per_var_metrics``.
    Saves to ``out_path`` (e.g. ``plots/per_var_metrics.png``).
    """
    per_var: Optional[Dict[str, Dict[str, float]]] = None
    source_label: str = ''

    # Prefer final_test
    if final_test is not None and final_test.get('per_var_metrics'):
        per_var = final_test['per_var_metrics']
        source_label = 'Final Test'
    else:
        # Fallback: last valid epoch with per_var_metrics
        for d in reversed(valid_data):
            if d.get('per_var_metrics'):
                per_var = d['per_var_metrics']
                source_label = f'Validation (epoch {d["epoch"]})'
                break

    if not per_var:
        return

    var_names = list(per_var.keys())
    rmse_vals = [per_var[v].get('rmse', 0.0) for v in var_names]
    mae_vals = [per_var[v].get('mae', 0.0) for v in var_names]

    n_vars = len(var_names)
    x = np.arange(n_vars)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_vars * 1.2), 5))

    bars_rmse = ax.bar(x - bar_width / 2, rmse_vals, bar_width,
                       color=COLOR_RMSE, alpha=0.80, label='RMSE')
    bars_mae = ax.bar(x + bar_width / 2, mae_vals, bar_width,
                      color=COLOR_MAE, alpha=0.80, label='MAE')

    # Value labels on bars
    for bar in bars_rmse:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7, color=COLOR_RMSE)
    for bar in bars_mae:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7, color=COLOR_MAE)

    ax.set_xticks(x)
    ax.set_xticklabels(var_names, fontsize=10, rotation=30 if n_vars > 6 else 0)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title(f'Per-Variable Metrics ({source_label})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_training_summary(
    data: Dict[str, Any],
    out_path: str,
) -> None:
    """
    Plot 5: Combined summary dashboard (2x2 grid + text panel).

    Layout:
      Row 0:  [Loss Curve]            [Metrics Curve]
      Row 1:  [LR Schedule]           [Summary Text]

    Saves to ``out_path`` (e.g. ``plots/training_summary.png``).
    """
    train_data = data['epochs_train']
    valid_data = data['epochs_valid']
    final_test = data.get('final_test')
    training_start = data.get('training_start')
    training_end = data.get('training_end')

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # ---- Panel (0, 0): Loss curve ----
    ax_loss = fig.add_subplot(gs[0, 0])
    if train_data:
        t_epochs = [d['epoch'] for d in train_data]
        t_loss = [d['train_loss'] for d in train_data]
        ax_loss.plot(t_epochs, t_loss, color=COLOR_TRAIN_LOSS, linewidth=1.2,
                     label='Train', alpha=0.85)
    if valid_data:
        v_epochs = [d['epoch'] for d in valid_data]
        v_loss = [d['valid_loss'] for d in valid_data]
        ax_loss.plot(v_epochs, v_loss, color=COLOR_VALID_LOSS, linewidth=1.2,
                     marker='o', markersize=2, label='Valid', alpha=0.85)
    ax_loss.set_xlabel('Epoch', fontsize=9)
    ax_loss.set_ylabel('Loss', fontsize=9)
    ax_loss.set_title('Loss Curve', fontsize=11, fontweight='bold')
    ax_loss.legend(fontsize=8)
    ax_loss.tick_params(labelsize=8)

    # ---- Panel (0, 1): RMSE / MAE ----
    ax_met = fig.add_subplot(gs[0, 1])
    if valid_data:
        m_epochs: List[int] = []
        m_rmse: List[float] = []
        m_mae: List[float] = []
        for d in valid_data:
            metrics = d.get('metrics', {})
            r = metrics.get('rmse')
            m = metrics.get('mae')
            if r is not None or m is not None:
                m_epochs.append(d['epoch'])
                m_rmse.append(float(r) if r is not None else float('nan'))
                m_mae.append(float(m) if m is not None else float('nan'))
        if m_epochs:
            ax_met.plot(m_epochs, m_rmse, color=COLOR_RMSE, linewidth=1.2,
                        marker='s', markersize=2, label='RMSE', alpha=0.85)
            ax_met_twin = ax_met.twinx()
            ax_met_twin.plot(m_epochs, m_mae, color=COLOR_MAE, linewidth=1.2,
                             marker='^', markersize=2, label='MAE', alpha=0.85)
            ax_met_twin.set_ylabel('MAE', fontsize=9, color=COLOR_MAE)
            ax_met_twin.tick_params(axis='y', labelcolor=COLOR_MAE, labelsize=8)

            # Combined legend
            lines_a, labels_a = ax_met.get_legend_handles_labels()
            lines_b, labels_b = ax_met_twin.get_legend_handles_labels()
            ax_met.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc='upper right')
    ax_met.set_xlabel('Epoch', fontsize=9)
    ax_met.set_ylabel('RMSE', fontsize=9, color=COLOR_RMSE)
    ax_met.set_title('Validation Metrics', fontsize=11, fontweight='bold')
    ax_met.tick_params(axis='y', labelcolor=COLOR_RMSE, labelsize=8)
    ax_met.tick_params(axis='x', labelsize=8)

    # ---- Panel (1, 0): LR schedule ----
    ax_lr = fig.add_subplot(gs[1, 0])
    if train_data:
        lr_epochs = [d['epoch'] for d in train_data]
        lr_vals = [d.get('lr', 0.0) for d in train_data]
        ax_lr.plot(lr_epochs, lr_vals, color=COLOR_LR, linewidth=1.2, alpha=0.85)
        ax_lr.fill_between(lr_epochs, 0, lr_vals, color=COLOR_LR, alpha=0.10)
    ax_lr.set_xlabel('Epoch', fontsize=9)
    ax_lr.set_ylabel('Learning Rate', fontsize=9)
    ax_lr.set_title('LR Schedule', fontsize=11, fontweight='bold')
    ax_lr.tick_params(labelsize=8)
    ax_lr.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    # ---- Panel (1, 1): Summary text ----
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis('off')

    summary_lines: List[str] = []
    summary_lines.append('--- Training Summary ---')
    summary_lines.append('')

    if training_start:
        summary_lines.append(f"Model:   {training_start.get('model_name', 'N/A')}")
        summary_lines.append(f"Device:  {training_start.get('device', 'N/A')}")
        summary_lines.append(f"Epochs:  {training_start.get('epochs', 'N/A')}")
        summary_lines.append('')

    if training_end:
        summary_lines.append(f"Best Epoch:    {training_end.get('best_epoch', 'N/A')}")
        summary_lines.append(f"Total Epochs:  {training_end.get('total_epochs', 'N/A')}")
        fm = training_end.get('final_metrics', {})
        if fm:
            summary_lines.append('')
            summary_lines.append('Final Metrics:')
            for k, v in fm.items():
                if isinstance(v, float):
                    summary_lines.append(f"  {k}: {v:.6f}")
                else:
                    summary_lines.append(f"  {k}: {v}")
        summary_lines.append('')

    if final_test:
        test_metrics = final_test.get('metrics', {})
        if test_metrics:
            summary_lines.append('Test Metrics:')
            for k, v in test_metrics.items():
                if isinstance(v, float):
                    summary_lines.append(f"  {k}: {v:.6f}")
                else:
                    summary_lines.append(f"  {k}: {v}")

        per_var = final_test.get('per_var_metrics')
        if per_var:
            summary_lines.append('')
            summary_lines.append('Per-Variable (Test):')
            for vname, vm in per_var.items():
                rmse_v = vm.get('rmse', 'N/A')
                mae_v = vm.get('mae', 'N/A')
                if isinstance(rmse_v, float):
                    rmse_v = f'{rmse_v:.6f}'
                if isinstance(mae_v, float):
                    mae_v = f'{mae_v:.6f}'
                summary_lines.append(f"  {vname}: RMSE={rmse_v}, MAE={mae_v}")

    if not summary_lines or len(summary_lines) <= 2:
        summary_lines.append('No summary data available.')

    text_content = '\n'.join(summary_lines)
    ax_txt.text(
        0.05, 0.95, text_content,
        transform=ax_txt.transAxes,
        fontsize=9, fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#cccccc'),
    )

    fig.suptitle('Ocean Forecast Training Summary', fontsize=15, fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =========================================================================
# 3. Main entry point
# =========================================================================

def generate_all_plots(log_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse training logs in *log_dir*, generate all five plots, and return a
    result dict.

    Parameters
    ----------
    log_dir : str
        Directory containing the training log file(s).
    output_dir : str, optional
        Where to write PNGs. Defaults to ``{log_dir}/plots/``.

    Returns
    -------
    dict
        ``{"status": "success"|"error", "plots": [...], "warnings": [...], ...}``
    """
    if output_dir is None:
        output_dir = os.path.join(log_dir, 'plots')

    result: Dict[str, Any] = {
        'status': 'success',
        'log_dir': log_dir,
        'output_dir': output_dir,
        'plots': [],
        'warnings': [],
        'errors': [],
    }

    # ------------------------------------------------------------------
    # Parse events
    # ------------------------------------------------------------------
    if not os.path.isdir(log_dir):
        result['status'] = 'error'
        result['errors'].append(f'log_dir does not exist: {log_dir}')
        return result

    events = parse_log_file(log_dir)
    if not events:
        result['status'] = 'error'
        result['errors'].append(
            f'No __event__{{...}}__event__ markers found in any file under {log_dir}'
        )
        return result

    data = extract_training_data(events)
    train_data = data['epochs_train']
    valid_data = data['epochs_valid']
    final_test = data.get('final_test')

    result['event_counts'] = {
        'epoch_train': len(train_data),
        'epoch_valid': len(valid_data),
        'has_final_test': final_test is not None,
    }

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: Loss curve
    # ------------------------------------------------------------------
    if train_data or valid_data:
        path = os.path.join(output_dir, 'loss_curve.png')
        try:
            plot_loss_curve(train_data, valid_data, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate loss_curve.png: {exc}')
    else:
        result['warnings'].append('No epoch_train / epoch_valid events; skipping loss_curve.png')

    # ------------------------------------------------------------------
    # Plot 2: Metrics curve (RMSE / MAE)
    # ------------------------------------------------------------------
    if valid_data:
        path = os.path.join(output_dir, 'metrics_curve.png')
        try:
            plot_metrics_curve(valid_data, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate metrics_curve.png: {exc}')
    else:
        result['warnings'].append('No epoch_valid events; skipping metrics_curve.png')

    # ------------------------------------------------------------------
    # Plot 3: LR schedule
    # ------------------------------------------------------------------
    if train_data:
        path = os.path.join(output_dir, 'lr_curve.png')
        try:
            plot_lr_curve(train_data, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate lr_curve.png: {exc}')
    else:
        result['warnings'].append('No epoch_train events; skipping lr_curve.png')

    # ------------------------------------------------------------------
    # Plot 4: Per-variable metrics bar chart
    # ------------------------------------------------------------------
    has_per_var = (
        (final_test is not None and final_test.get('per_var_metrics'))
        or any(d.get('per_var_metrics') for d in valid_data)
    )
    if has_per_var:
        path = os.path.join(output_dir, 'per_var_metrics.png')
        try:
            plot_per_var_metrics(valid_data, final_test, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate per_var_metrics.png: {exc}')
    else:
        result['warnings'].append('No per_var_metrics data; skipping per_var_metrics.png')

    # ------------------------------------------------------------------
    # Plot 5: Combined training summary
    # ------------------------------------------------------------------
    if train_data or valid_data:
        path = os.path.join(output_dir, 'training_summary.png')
        try:
            plot_training_summary(data, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate training_summary.png: {exc}')
    else:
        result['warnings'].append('Insufficient data; skipping training_summary.png')

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------
    if not result['plots']:
        result['status'] = 'error'
        result['errors'].append('No plots were generated.')
    else:
        result['message'] = f"Generated {len(result['plots'])} plot(s) in {output_dir}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate training visualization plots for ocean forecast models.'
    )
    parser.add_argument(
        '--log_dir', required=True,
        help='Directory containing training log files with __event__ markers.',
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory for plots. Defaults to {log_dir}/plots/.',
    )
    args = parser.parse_args()

    result = generate_all_plots(args.log_dir, args.output_dir)

    # Emit structured result for the TypeScript process manager
    result_json = json.dumps(result, ensure_ascii=False, default=str)
    print(f'__result__{result_json}__result__')


if __name__ == '__main__':
    main()
