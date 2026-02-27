#!/usr/bin/env python3
"""
@file generate_training_plots.py
@description Generate training visualization plots for ocean forecast models.
             Visual style aligned with ocean_SR_training_masked/training_plot_lib/.
@author Leizheng
@date 2026-02-26
@version 2.1.0

@changelog
  - 2026-02-27 Leizheng: v2.1.0 fix duplicate epoch data from DDP/multi-log,
        reduce fill alpha, skip fill when LR constant, fix Test Loss card color
  - 2026-02-27 Leizheng: v2.0.0 rewrite with SR-style visuals (modern palette,
        sparse markers, fill-under, star-best, card-based summary, figure borders)
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
"""

import argparse
import json
import math
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as patheffects
except ImportError:
    print(json.dumps({
        "status": "error",
        "errors": ["matplotlib is required: pip install matplotlib"]
    }))
    sys.exit(1)

# =========================================================================
# Style system  (mirrors ocean_SR_training_masked/training_plot_lib/style.py)
# =========================================================================

def _detect_cjk_font() -> Optional[str]:
    candidates = [
        'SimHei', 'SimSun', 'Microsoft YaHei',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans CN',
        'PingFang SC', 'STSong', 'AR PL UMing CN',
        'Droid Sans Fallback',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return None


_cjk_font = _detect_cjk_font()
_sans_serif_list = ['DejaVu Sans', _cjk_font] if _cjk_font else ['DejaVu Sans']

for _s in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid']:
    if _s in plt.style.available:
        plt.style.use(_s)
        break

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': _sans_serif_list,
    'font.size': 11,
    'axes.unicode_minus': False,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'medium',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#ffffff',
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'savefig.facecolor': '#fafafa',
    'savefig.edgecolor': 'none',
})

warnings.filterwarnings('ignore', message='Glyph .* missing from font')

# Color palette
COLORS = {
    'primary': '#2563eb',
    'secondary': '#dc2626',
    'success': '#16a34a',
    'warning': '#ea580c',
    'purple': '#9333ea',
    'cyan': '#0891b2',
    'pink': '#db2777',
    'gray': '#6b7280',
}

DPI = 180


def add_figure_border(fig, color='#e5e7eb', linewidth=2, padding=0.02):
    rect = FancyBboxPatch(
        (padding, padding), 1 - 2 * padding, 1 - 2 * padding,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(rect)


def calc_marker_interval(n_points: int, target_markers: int = 15) -> int:
    if n_points <= target_markers:
        return 1
    return max(1, n_points // target_markers)


# =========================================================================
# 1. Log parsing
# =========================================================================

_EVENT_RE = re.compile(r'__event__(\{.*?\})__event__')


def parse_log_file(log_dir: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    log_files: List[str] = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith('.log') or fname.endswith('.txt'):
            log_files.append(os.path.join(log_dir, fname))

    if not log_files:
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


def _dedup_by_epoch(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate epoch records (DDP multi-rank / duplicate log files).

    When multiple records share the same epoch number, keep only the first
    occurrence and discard the rest.  Result is sorted by epoch ascending.
    """
    seen: Dict[int, Dict[str, Any]] = {}
    for rec in records:
        ep = rec.get('epoch')
        if ep is not None and ep not in seen:
            seen[ep] = rec
    return [seen[k] for k in sorted(seen)]


def extract_training_data(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    epochs_train: List[Dict[str, Any]] = []
    epochs_valid: List[Dict[str, Any]] = []
    final_test: Optional[Dict[str, Any]] = None
    training_start: Optional[Dict[str, Any]] = None
    training_end: Optional[Dict[str, Any]] = None

    for ev in events:
        etype = ev.get('event', '') or ev.get('type', '')
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
        'epochs_train': _dedup_by_epoch(epochs_train),
        'epochs_valid': _dedup_by_epoch(epochs_valid),
        'final_test': final_test,
        'training_start': training_start,
        'training_end': training_end,
    }


# =========================================================================
# 2. Individual plot functions  (SR-aligned style)
# =========================================================================

def plot_loss_curve(
    train_data: List[Dict[str, Any]],
    valid_data: List[Dict[str, Any]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    # Train loss
    if train_data:
        t_epochs = [d['epoch'] for d in train_data]
        t_loss = [d['train_loss'] for d in train_data]
        mi = calc_marker_interval(len(t_epochs))

        ax.plot(t_epochs, t_loss, color=COLORS['primary'], linewidth=2.5,
                label='Train Loss', zorder=3)
        ax.scatter(t_epochs[::mi], t_loss[::mi],
                   color=COLORS['primary'], s=40,
                   facecolors='white', edgecolors=COLORS['primary'],
                   linewidths=2, zorder=4)
        ax.fill_between(t_epochs, t_loss, alpha=0.10, color=COLORS['primary'],
                        edgecolor='none')

    # Valid loss
    if valid_data:
        v_epochs = [d['epoch'] for d in valid_data]
        v_loss = [d['valid_loss'] for d in valid_data]
        mi = calc_marker_interval(len(v_epochs))

        ax.plot(v_epochs, v_loss, color=COLORS['secondary'], linewidth=2.5,
                label='Valid Loss', zorder=3)
        ax.scatter(v_epochs[::mi], v_loss[::mi],
                   color=COLORS['secondary'], s=40, marker='s',
                   facecolors='white', edgecolors=COLORS['secondary'],
                   linewidths=2, zorder=4)
        ax.fill_between(v_epochs, v_loss, alpha=0.10, color=COLORS['secondary'],
                        edgecolor='none')

        # Best epoch annotation
        if v_loss:
            best_idx = int(np.argmin(v_loss))
            best_epoch = v_epochs[best_idx]
            best_loss = v_loss[best_idx]

            ax.axvline(x=best_epoch, color=COLORS['success'], linestyle='--',
                       alpha=0.6, linewidth=1.5)
            ax.scatter([best_epoch], [best_loss], s=150, color=COLORS['success'],
                       marker='*', zorder=5, edgecolors='white', linewidths=1.5)

            # Smart positioning
            x_range = max(v_epochs) - min(v_epochs)
            y_range = max(v_loss) - min(v_loss) if max(v_loss) != min(v_loss) else 1
            text_x = -80 if best_epoch > min(v_epochs) + x_range * 0.5 else 20
            text_y = -40 if best_loss > min(v_loss) + y_range * 0.5 else 30

            ax.annotate(
                f'Best: {best_loss:.6f}\nEpoch {best_epoch}',
                xy=(best_epoch, best_loss),
                xytext=(text_x, text_y), textcoords='offset points',
                fontsize=10, fontweight='medium', color=COLORS['success'],
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                          edgecolor=COLORS['success'], alpha=0.95, linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'],
                                connectionstyle='arc3,rad=0.2', linewidth=1.5),
            )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve', pad=15)

    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_linewidth(1.2)

    add_figure_border(fig)
    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_curve(
    valid_data: List[Dict[str, Any]],
    out_path: str,
) -> None:
    if not valid_data:
        return

    epochs: List[int] = []
    rmse_vals: List[float] = []
    mae_vals: List[float] = []

    for d in valid_data:
        metrics = d.get('metrics', {})
        r = metrics.get('rmse')
        m = metrics.get('mae')
        if r is not None or m is not None:
            epochs.append(d['epoch'])
            rmse_vals.append(float(r) if r is not None else float('nan'))
            mae_vals.append(float(m) if m is not None else float('nan'))

    if not epochs:
        return

    has_rmse = any(math.isfinite(v) for v in rmse_vals)
    has_mae = any(math.isfinite(v) for v in mae_vals)

    # 2-panel layout (like SR metrics: each metric gets its own subplot)
    n_panels = int(has_rmse) + int(has_mae)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # RMSE panel
    if has_rmse:
        ax = axes[panel_idx]
        panel_idx += 1
        mi = calc_marker_interval(len(epochs))
        ax.plot(epochs, rmse_vals, color=COLORS['warning'], linewidth=2.5, zorder=3)
        ax.scatter(epochs[::mi], rmse_vals[::mi],
                   color=COLORS['warning'], s=36,
                   facecolors='white', edgecolors=COLORS['warning'],
                   linewidths=2, zorder=4)
        ax.fill_between(epochs, rmse_vals, alpha=0.10, color=COLORS['warning'],
                        edgecolor='none')

        # Best RMSE
        finite = [(i, v) for i, v in enumerate(rmse_vals) if math.isfinite(v)]
        if finite:
            best_i, best_v = min(finite, key=lambda x: x[1])
            ax.axhline(y=best_v, color=COLORS['warning'], linestyle=':', alpha=0.5, linewidth=1.5)
            ax.scatter([epochs[best_i]], [best_v], s=180, color='#fbbf24',
                       marker='*', zorder=5, edgecolors=COLORS['warning'], linewidths=1.5)
            ax.annotate(
                f'Best: {best_v:.4f}\n(Epoch {epochs[best_i]})',
                xy=(epochs[best_i], best_v),
                xytext=(15, 25), textcoords='offset points',
                fontsize=9, fontweight='medium', color=COLORS['warning'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=COLORS['warning'], alpha=0.95, linewidth=1.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'],
                                connectionstyle='arc3,rad=0.2', linewidth=1.2),
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Validation RMSE', pad=12)

    # MAE panel
    if has_mae:
        ax = axes[panel_idx]
        mi = calc_marker_interval(len(epochs))
        ax.plot(epochs, mae_vals, color=COLORS['primary'], linewidth=2.5, zorder=3)
        ax.scatter(epochs[::mi], mae_vals[::mi],
                   color=COLORS['primary'], s=36,
                   facecolors='white', edgecolors=COLORS['primary'],
                   linewidths=2, zorder=4)
        ax.fill_between(epochs, mae_vals, alpha=0.10, color=COLORS['primary'],
                        edgecolor='none')

        # Best MAE
        finite = [(i, v) for i, v in enumerate(mae_vals) if math.isfinite(v)]
        if finite:
            best_i, best_v = min(finite, key=lambda x: x[1])
            ax.axhline(y=best_v, color=COLORS['primary'], linestyle=':', alpha=0.5, linewidth=1.5)
            ax.scatter([epochs[best_i]], [best_v], s=180, color='#fbbf24',
                       marker='*', zorder=5, edgecolors=COLORS['primary'], linewidths=1.5)
            ax.annotate(
                f'Best: {best_v:.4f}\n(Epoch {epochs[best_i]})',
                xy=(epochs[best_i], best_v),
                xytext=(15, 25), textcoords='offset points',
                fontsize=9, fontweight='medium', color=COLORS['primary'],
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=COLORS['primary'], alpha=0.95, linewidth=1.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'],
                                connectionstyle='arc3,rad=0.2', linewidth=1.2),
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Validation MAE', pad=12)

    add_figure_border(fig)
    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_lr_curve(
    train_data: List[Dict[str, Any]],
    out_path: str,
) -> None:
    if not train_data:
        return

    epochs = [d['epoch'] for d in train_data]
    lrs = [d.get('lr', 0.0) for d in train_data]

    if not any(lr > 0 for lr in lrs):
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    mi = calc_marker_interval(len(epochs))

    ax.plot(epochs, lrs, color=COLORS['success'], linewidth=2.5, zorder=3)
    ax.scatter(epochs[::mi], lrs[::mi],
               color=COLORS['success'], s=40,
               facecolors='white', edgecolors=COLORS['success'],
               linewidths=2, zorder=4)

    # Only fill when LR actually changes (skip for constant LR)
    lr_range = max(lrs) - min(lrs) if lrs else 0
    if lr_range > min(lrs) * 0.01:
        ax.fill_between(epochs, 0, lrs, alpha=0.10, color=COLORS['success'],
                        edgecolor='none')

    # Start / End LR annotations
    if len(lrs) >= 2:
        ax.annotate(
            f'Start: {lrs[0]:.2e}',
            xy=(epochs[0], lrs[0]),
            xytext=(10, 15), textcoords='offset points',
            fontsize=9, fontweight='medium', color=COLORS['primary'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                      edgecolor=COLORS['primary'], alpha=0.95, linewidth=1.2),
        )
        ax.annotate(
            f'End: {lrs[-1]:.2e}',
            xy=(epochs[-1], lrs[-1]),
            xytext=(-60, 15), textcoords='offset points',
            fontsize=9, fontweight='medium', color=COLORS['secondary'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                      edgecolor=COLORS['secondary'], alpha=0.95, linewidth=1.2),
        )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', pad=15)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    add_figure_border(fig)
    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_per_var_metrics(
    valid_data: List[Dict[str, Any]],
    final_test: Optional[Dict[str, Any]],
    out_path: str,
) -> None:
    per_var: Optional[Dict[str, Dict[str, float]]] = None
    source_label: str = ''

    if final_test is not None and final_test.get('per_var_metrics'):
        per_var = final_test['per_var_metrics']
        source_label = 'Final Test'
    else:
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

    fig, ax = plt.subplots(figsize=(max(8, n_vars * 2.0), 5))

    # Bars with white edge (SR style)
    bars_rmse = ax.bar(x - bar_width / 2, rmse_vals, bar_width,
                       color=COLORS['warning'], alpha=0.85, label='RMSE',
                       edgecolor='white', linewidth=2)
    bars_mae = ax.bar(x + bar_width / 2, mae_vals, bar_width,
                      color=COLORS['primary'], alpha=0.85, label='MAE',
                      edgecolor='white', linewidth=2)

    # Value labels with white stroke
    for bar, color in [(bars_rmse, COLORS['warning']), (bars_mae, COLORS['primary'])]:
        for b in bar:
            h = b.get_height()
            txt = ax.text(b.get_x() + b.get_width() / 2, h,
                          f'{h:.4f}', ha='center', va='bottom',
                          fontsize=9, fontweight='bold', color=color)
            txt.set_path_effects([
                patheffects.withStroke(linewidth=3, foreground='white'),
            ])

    ax.set_xticks(x)
    ax.set_xticklabels(var_names, fontsize=11, rotation=30 if n_vars > 6 else 0)
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Per-Variable Metrics ({source_label})', pad=15)

    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                       ncol=2)
    legend.get_frame().set_linewidth(1.2)

    add_figure_border(fig)
    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_training_summary(
    data: Dict[str, Any],
    out_path: str,
) -> None:
    """Card-based summary dashboard (mirrors SR plot_summary.py)."""
    training_start = data.get('training_start') or {}
    training_end = data.get('training_end') or {}
    final_test = data.get('final_test') or {}

    if not training_end and not training_start:
        return

    # Extract data
    model_name = training_start.get('model_name', 'N/A')
    dataset_name = training_start.get('dataset_name', 'N/A')
    model_params = training_start.get('model_params', 'N/A')
    total_epochs = training_start.get('total_epochs', 'N/A')
    actual_epochs = training_end.get('actual_epochs', training_end.get('total_epochs', 'N/A'))
    best_epoch = training_end.get('best_epoch', 'N/A')
    duration = training_end.get('training_duration_seconds', 0)
    test_metrics = final_test.get('metrics', {})

    # Format duration
    if duration < 60:
        duration_str = f"{duration:.1f}s"
    elif duration < 3600:
        duration_str = f"{duration / 60:.1f} min"
    else:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours}h {minutes}min"

    params_str = f"{model_params}M" if isinstance(model_params, (int, float)) else str(model_params)
    early_stopped = actual_epochs != total_epochs if isinstance(actual_epochs, int) and isinstance(total_epochs, int) else False

    # Card drawing helpers
    def draw_card(ax, title, items, title_color=COLORS['primary']):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        card_bg = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor='white', edgecolor='#e2e8f0', linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(card_bg)

        title_bg = FancyBboxPatch(
            (0.02, 0.82), 0.96, 0.16,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=title_color, edgecolor='none',
            transform=ax.transAxes,
        )
        ax.add_patch(title_bg)

        ax.text(0.5, 0.90, title, fontsize=13, fontweight='bold',
                ha='center', va='center', color='white', transform=ax.transAxes)

        n_items = len(items)
        if n_items == 0:
            return
        content_top = 0.78
        content_bottom = 0.08
        content_height = content_top - content_bottom
        row_height = content_height / n_items

        for i, (label, value) in enumerate(items):
            y_pos = content_top - (i + 0.5) * row_height
            if i % 2 == 0:
                row_bg = FancyBboxPatch(
                    (0.04, y_pos - row_height * 0.45), 0.92, row_height * 0.9,
                    boxstyle="round,pad=0.005,rounding_size=0.01",
                    facecolor='#f8fafc', edgecolor='none',
                    transform=ax.transAxes,
                )
                ax.add_patch(row_bg)

            ax.text(0.08, y_pos, label, fontsize=10, fontweight='medium',
                    ha='left', va='center', color='#64748b', transform=ax.transAxes)
            ax.text(0.92, y_pos, str(value), fontsize=10, fontweight='semibold',
                    ha='right', va='center', color='#1e293b', transform=ax.transAxes)

    def draw_metric_card(ax, label, value, color, better='lower'):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        card_bg = FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor='white', edgecolor=color, linewidth=2,
            transform=ax.transAxes,
        )
        ax.add_patch(card_bg)

        top_bar = FancyBboxPatch(
            (0.05, 0.85), 0.9, 0.1,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=color, edgecolor='none',
            transform=ax.transAxes,
        )
        ax.add_patch(top_bar)

        ax.text(0.5, 0.68, label, fontsize=11, fontweight='bold',
                ha='center', va='center', color='#475569', transform=ax.transAxes)

        if value != 'N/A':
            ax.text(0.5, 0.40, value, fontsize=16, fontweight='bold',
                    ha='center', va='center', color=color, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.40, 'N/A', fontsize=14, fontweight='medium',
                    ha='center', va='center', color='#94a3b8', style='italic',
                    transform=ax.transAxes)

        direction_text = '(lower is better)' if better == 'lower' else '(higher is better)'
        ax.text(0.5, 0.18, direction_text, fontsize=8,
                ha='center', va='center', color='#94a3b8', transform=ax.transAxes)

    # Build figure
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('#f8fafc')

    fig.text(0.5, 0.95, 'Training Summary', fontsize=22, fontweight='bold',
             ha='center', va='top', color='#1e293b')

    # Upper half: two info cards
    ax_info1 = fig.add_axes([0.04, 0.48, 0.44, 0.42])
    ax_info2 = fig.add_axes([0.52, 0.48, 0.44, 0.42])

    draw_card(ax_info1, 'Model Information', [
        ('Model', model_name),
        ('Dataset', dataset_name),
        ('Parameters', params_str),
    ], COLORS['primary'])

    draw_card(ax_info2, 'Training Progress', [
        ('Planned Epochs', str(total_epochs)),
        ('Actual Epochs', str(actual_epochs)),
        ('Best Epoch', str(best_epoch)),
        ('Early Stopped', 'Yes' if early_stopped else 'No'),
        ('Duration', duration_str),
    ], COLORS['success'])

    # Lower half: metric cards
    # Forecast uses MSE/RMSE/MAE (no PSNR/SSIM)
    mse_val = f"{test_metrics.get('mse'):.6f}" if isinstance(test_metrics.get('mse'), float) else 'N/A'
    rmse_val = f"{test_metrics.get('rmse'):.6f}" if isinstance(test_metrics.get('rmse'), float) else 'N/A'
    mae_val = f"{test_metrics.get('mae'):.6f}" if isinstance(test_metrics.get('mae'), float) else 'N/A'
    loss_val = f"{test_metrics.get('test_loss'):.6f}" if isinstance(test_metrics.get('test_loss'), float) else 'N/A'

    ax_m1 = fig.add_axes([0.04, 0.06, 0.22, 0.36])
    ax_m2 = fig.add_axes([0.28, 0.06, 0.22, 0.36])
    ax_m3 = fig.add_axes([0.52, 0.06, 0.22, 0.36])
    ax_m4 = fig.add_axes([0.76, 0.06, 0.22, 0.36])

    draw_metric_card(ax_m1, 'Test Loss', loss_val, COLORS['purple'], 'lower')
    draw_metric_card(ax_m2, 'Test MSE', mse_val, COLORS['primary'], 'lower')
    draw_metric_card(ax_m3, 'Test RMSE', rmse_val, COLORS['warning'], 'lower')
    draw_metric_card(ax_m4, 'Test MAE', mae_val, COLORS['success'], 'lower')

    add_figure_border(fig, color='#cbd5e1', linewidth=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='#f8fafc')
    plt.close(fig)


# =========================================================================
# 3. Main entry point
# =========================================================================

def generate_all_plots(log_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
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

    # Plot 1: Loss curve
    if train_data or valid_data:
        path = os.path.join(output_dir, 'loss_curve.png')
        try:
            plot_loss_curve(train_data, valid_data, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate loss_curve.png: {exc}')

    # Plot 2: Metrics curve (RMSE / MAE)
    if valid_data:
        path = os.path.join(output_dir, 'metrics_curve.png')
        try:
            plot_metrics_curve(valid_data, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate metrics_curve.png: {exc}')

    # Plot 3: LR schedule
    if train_data:
        path = os.path.join(output_dir, 'lr_curve.png')
        try:
            plot_lr_curve(train_data, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate lr_curve.png: {exc}')

    # Plot 4: Per-variable metrics bar chart
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

    # Plot 5: Card-based training summary
    if train_data or valid_data:
        path = os.path.join(output_dir, 'training_summary.png')
        try:
            plot_training_summary(data, path)
            result['plots'].append(path)
        except Exception as exc:
            result['warnings'].append(f'Failed to generate training_summary.png: {exc}')

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

    result_json = json.dumps(result, ensure_ascii=False, default=str)
    print(f'__result__{result_json}__result__')


if __name__ == '__main__':
    main()
