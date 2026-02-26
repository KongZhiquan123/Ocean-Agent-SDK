#!/usr/bin/env python3
"""
@file generate_predict_plots.py
@description Generate prediction visualization plots for ocean forecast models.
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast prediction visualization
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Chinese font support
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Colormap: prefer cmocean if available, fallback to viridis
try:
    import cmocean
    DEFAULT_CMAP = cmocean.cm.thermal
    ERROR_CMAP = cmocean.cm.amp
except ImportError:
    DEFAULT_CMAP = "viridis"
    ERROR_CMAP = "inferno"


# ============================================================================
# Prediction file parsing
# ============================================================================

# Expected filename pattern produced by trainers/base.py predict():
#   sample_{i:06d}_t{t}_var{c}_{var_name}.npy
_PRED_PATTERN = re.compile(
    r"^sample_(?P<sample>\d+)_t(?P<timestep>\d+)_var(?P<varidx>\d+)_(?P<varname>.+)\.npy$"
)


def _parse_pred_filename(fname: str) -> Optional[Dict[str, Any]]:
    """Parse a prediction NPY filename into its components.

    Returns:
        Dict with keys: sample (int), timestep (int), varidx (int), varname (str),
        or None if the filename does not match the expected pattern.
    """
    m = _PRED_PATTERN.match(fname)
    if m is None:
        return None
    return {
        "sample": int(m.group("sample")),
        "timestep": int(m.group("timestep")),
        "varidx": int(m.group("varidx")),
        "varname": m.group("varname"),
    }


def discover_predictions(pred_dir: str) -> Dict[int, Dict[int, Dict[int, Dict[str, Any]]]]:
    """Scan the predictions directory and build an index.

    Returns:
        Nested dict:  sample_idx -> timestep -> var_idx -> {
            "varname": str, "path": str
        }
    """
    index: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    if not os.path.isdir(pred_dir):
        return index

    for fname in sorted(os.listdir(pred_dir)):
        parsed = _parse_pred_filename(fname)
        if parsed is None:
            continue
        index[parsed["sample"]][parsed["timestep"]][parsed["varidx"]] = {
            "varname": parsed["varname"],
            "path": os.path.join(pred_dir, fname),
        }
    return index


def get_var_names_from_index(
    index: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]]
) -> List[str]:
    """Extract an ordered list of variable names from the prediction index."""
    var_map: Dict[int, str] = {}
    for sample_data in index.values():
        for ts_data in sample_data.values():
            for vidx, vinfo in ts_data.items():
                if vidx not in var_map:
                    var_map[vidx] = vinfo["varname"]
    return [var_map[k] for k in sorted(var_map.keys())]


# ============================================================================
# Ground truth loading
# ============================================================================

def load_ground_truth_test_set(
    dataset_root: str,
    dyn_vars: List[str],
    in_t: int,
    out_t: int,
    stride: int,
) -> Optional[np.ndarray]:
    """Load ground truth target data from the test split of the dataset.

    The dataset is organized as:
        dataset_root/test/{var_name}/{date_str}.npy   -- each (H, W)

    We replicate the sliding-window logic from OceanForecastNpyDataset to
    extract the *target* frames for each sample so they can be compared
    against predictions.

    Returns:
        Array of shape (N_samples, out_t, n_vars, H, W), or None on failure.
    """
    if dataset_root is None:
        return None

    # Read time index to get filenames
    time_index_path = os.path.join(dataset_root, "time_index.json")
    if not os.path.isfile(time_index_path):
        return None

    try:
        with open(time_index_path, "r", encoding="utf-8") as f:
            time_index_cfg = json.load(f)
    except Exception:
        return None

    split_info = time_index_cfg.get("splits", {}).get("test", {})
    filenames = split_info.get("filenames", split_info.get("timestamps", []))
    if not filenames:
        return None

    # Load the raw data tensor (T, H, W, C)
    T = len(filenames)
    C = len(dyn_vars)

    # Determine spatial shape from the first available file
    H = W = None
    for fname in filenames:
        for var_name in dyn_vars:
            npy_path = os.path.join(dataset_root, "test", var_name, f"{fname}.npy")
            if os.path.isfile(npy_path):
                arr = np.load(npy_path)
                if arr.ndim == 2:
                    H, W = arr.shape
                elif arr.ndim == 3:
                    H, W = arr.shape[1], arr.shape[2]
                break
        if H is not None:
            break

    if H is None or W is None:
        return None

    # Build the full tensor
    raw = np.zeros((T, H, W, C), dtype=np.float32)
    for t_idx, fname in enumerate(filenames):
        for c_idx, var_name in enumerate(dyn_vars):
            npy_path = os.path.join(dataset_root, "test", var_name, f"{fname}.npy")
            if not os.path.isfile(npy_path):
                continue
            arr = np.load(npy_path).astype(np.float32)
            if arr.ndim == 2:
                raw[t_idx, :, :, c_idx] = arr
            elif arr.ndim == 3:
                raw[t_idx, :, :, c_idx] = arr[0]

    # Replace NaN with 0 (consistent with dataset class)
    raw = np.nan_to_num(raw, nan=0.0)

    # Sliding window: extract target frames (y)
    window_size = in_t + out_t
    if T < window_size:
        return None

    starts = list(range(0, T - window_size + 1, stride))
    if not starts:
        return None

    N = len(starts)
    # y_samples: (N, out_t, C, H, W)  -- rearranged for per-timestep/var access
    y_samples = np.empty((N, out_t, C, H, W), dtype=np.float32)
    for i, t0 in enumerate(starts):
        y_frames = raw[t0 + in_t : t0 + in_t + out_t]  # (out_t, H, W, C)
        # Transpose to (out_t, C, H, W)
        y_samples[i] = y_frames.transpose(0, 3, 1, 2)

    return y_samples


def load_ground_truth_config(dataset_root: str) -> Dict[str, Any]:
    """Load dataset configuration from var_names.json and config.yaml.

    Returns a dict with keys: dyn_vars, in_t, out_t, stride, spatial_shape.
    """
    result: Dict[str, Any] = {
        "dyn_vars": [],
        "in_t": 1,
        "out_t": 1,
        "stride": 1,
        "spatial_shape": None,
    }

    var_names_path = os.path.join(dataset_root, "var_names.json")
    if os.path.isfile(var_names_path):
        try:
            with open(var_names_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            result["dyn_vars"] = cfg.get("dynamic", cfg.get("dyn_vars", []))
            result["spatial_shape"] = cfg.get("spatial_shape")
        except Exception:
            pass

    return result


# ============================================================================
# Plotting functions
# ============================================================================

def _add_colorbar(fig: plt.Figure, ax: plt.Axes, im: Any) -> None:
    """Add a thin colorbar next to an axis without changing its size."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    fig.colorbar(im, cax=cax)


def plot_single_sample_var(
    pred: np.ndarray,
    truth: Optional[np.ndarray],
    var_name: str,
    sample_idx: int,
    timestep: int,
    output_path: str,
    dpi: int = 150,
) -> str:
    """Plot a 3-panel (or 1-panel) comparison for one sample/variable/timestep.

    Panels: Prediction | Ground Truth | Absolute Error
    If ground truth is unavailable, only the prediction panel is shown.

    Args:
        pred: 2D array (H, W) with prediction values.
        truth: 2D array (H, W) with ground truth values, or None.
        var_name: Variable name for the title.
        sample_idx: Sample index.
        timestep: Timestep index.
        output_path: Path to save the figure.
        dpi: Output DPI.

    Returns:
        The output_path on success.
    """
    has_truth = truth is not None

    if has_truth:
        n_cols = 3
        fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols + 0.5, 4.5), dpi=dpi)

        # Shared color scale for prediction and truth
        all_vals = np.concatenate([
            pred[np.isfinite(pred)].ravel(),
            truth[np.isfinite(truth)].ravel(),
        ])
        if all_vals.size > 0:
            vmin = float(np.nanpercentile(all_vals, 1))
            vmax = float(np.nanpercentile(all_vals, 99))
        else:
            vmin, vmax = 0.0, 1.0

        error = np.abs(pred - truth)
        emax = float(np.nanpercentile(error[np.isfinite(error)], 99)) if np.any(np.isfinite(error)) else 1.0

        # Panel 1: Prediction
        im0 = axes[0].imshow(pred, origin="lower", cmap=DEFAULT_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        axes[0].set_title("Prediction", fontsize=11, fontweight="bold")
        axes[0].axis("off")
        _add_colorbar(fig, axes[0], im0)

        # Panel 2: Ground Truth
        im1 = axes[1].imshow(truth, origin="lower", cmap=DEFAULT_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        axes[1].set_title("Ground Truth", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        _add_colorbar(fig, axes[1], im1)

        # Panel 3: Absolute Error
        im2 = axes[2].imshow(error, origin="lower", cmap=ERROR_CMAP, vmin=0, vmax=emax, aspect="auto")
        axes[2].set_title("|Pred - Truth|", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        _add_colorbar(fig, axes[2], im2)

    else:
        # No ground truth: single panel
        n_cols = 1
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=dpi)
        axes = [ax]

        finite = pred[np.isfinite(pred)]
        if finite.size > 0:
            vmin = float(np.nanpercentile(finite, 1))
            vmax = float(np.nanpercentile(finite, 99))
        else:
            vmin, vmax = 0.0, 1.0

        im0 = ax.imshow(pred, origin="lower", cmap=DEFAULT_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title("Prediction", fontsize=11, fontweight="bold")
        ax.axis("off")
        _add_colorbar(fig, ax, im0)

    fig.suptitle(
        f"Sample {sample_idx}  |  t={timestep}  |  {var_name}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_overview_grid(
    pred_index: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]],
    gt_data: Optional[np.ndarray],
    var_names: List[str],
    sample_indices: List[int],
    output_path: str,
    dpi: int = 120,
) -> str:
    """Generate an overview grid showing all visualized samples.

    Rows = samples, Columns = variables. Each cell shows the prediction heatmap.
    If ground truth is available, a small error annotation is overlaid.

    Args:
        pred_index: Prediction file index from discover_predictions().
        gt_data: Ground truth array (N, out_t, C, H, W) or None.
        var_names: List of variable names.
        sample_indices: List of sample indices to include.
        output_path: Output file path.
        dpi: Figure DPI.

    Returns:
        The output_path on success.
    """
    n_rows = len(sample_indices)
    n_cols = len(var_names)

    if n_rows == 0 or n_cols == 0:
        return output_path

    cell_w = 4.0
    cell_h = 3.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols + 1.0, cell_h * n_rows + 1.0),
        dpi=dpi,
        squeeze=False,
    )

    for row, si in enumerate(sample_indices):
        sample_data = pred_index.get(si, {})
        # Use timestep 0 for the overview
        ts_data = sample_data.get(0, {})

        for col, var_name in enumerate(var_names):
            ax = axes[row, col]
            var_info = ts_data.get(col)

            if var_info is None:
                ax.text(
                    0.5, 0.5, "N/A",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=12, color="gray",
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                pred = np.load(var_info["path"]).astype(np.float32)
                finite = pred[np.isfinite(pred)]
                if finite.size > 0:
                    vmin = float(np.nanpercentile(finite, 1))
                    vmax = float(np.nanpercentile(finite, 99))
                else:
                    vmin, vmax = 0.0, 1.0

                im = ax.imshow(
                    pred, origin="lower", cmap=DEFAULT_CMAP,
                    vmin=vmin, vmax=vmax, aspect="auto",
                )
                ax.axis("off")
                _add_colorbar(fig, ax, im)

                # Annotate with RMSE if ground truth is available
                if gt_data is not None and si < gt_data.shape[0]:
                    truth = gt_data[si, 0, col]  # timestep=0, var=col
                    rmse = float(np.sqrt(np.nanmean((pred - truth) ** 2)))
                    mae = float(np.nanmean(np.abs(pred - truth)))
                    ax.text(
                        0.02, 0.96,
                        f"RMSE={rmse:.4g}\nMAE={mae:.4g}",
                        transform=ax.transAxes,
                        fontsize=7,
                        color="white",
                        verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
                    )

            # Row / column labels
            if row == 0:
                ax.set_title(var_name, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"Sample {si}", fontsize=9, fontweight="medium")

    fig.suptitle(
        "Prediction Overview (t=0)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================================
# Main logic
# ============================================================================

def generate_predict_plots(
    log_dir: str,
    dataset_root: Optional[str] = None,
    n_samples: int = 5,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate prediction visualization plots.

    Args:
        log_dir: Training log directory containing predictions/ subdirectory.
        dataset_root: Optional path to preprocessed dataset for ground truth.
        n_samples: Number of samples to visualize.
        output_dir: Output directory for plots. Defaults to {log_dir}/plots/.

    Returns:
        Result dict with status, generated file list, and metadata.
    """
    result: Dict[str, Any] = {
        "status": "success",
        "log_dir": log_dir,
        "output_dir": output_dir,
        "plots": [],
        "warnings": [],
        "errors": [],
    }

    # Resolve paths
    pred_dir = os.path.join(log_dir, "predictions")
    if output_dir is None:
        output_dir = os.path.join(log_dir, "plots")
    result["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- Discover prediction files ----
    pred_index = discover_predictions(pred_dir)
    if not pred_index:
        result["status"] = "error"
        result["errors"].append(
            f"No prediction files found in {pred_dir}. "
            f"Expected format: sample_NNNNNN_tT_varC_VARNAME.npy"
        )
        return result

    all_sample_ids = sorted(pred_index.keys())
    var_names = get_var_names_from_index(pred_index)
    n_total = len(all_sample_ids)

    print(
        f"[Info] Found {n_total} samples, {len(var_names)} variables: {var_names}",
        file=sys.stderr,
    )

    # ---- Select samples to visualize ----
    n_vis = min(n_samples, n_total)
    if n_vis <= 0:
        result["status"] = "error"
        result["errors"].append("No samples to visualize.")
        return result

    # Uniformly sample from available samples
    if n_vis >= n_total:
        vis_sample_ids = all_sample_ids
    else:
        vis_indices = [int(i * (n_total - 1) / max(n_vis - 1, 1)) for i in range(n_vis)]
        vis_sample_ids = [all_sample_ids[i] for i in vis_indices]

    # ---- Load ground truth (optional) ----
    gt_data: Optional[np.ndarray] = None
    if dataset_root is not None and os.path.isdir(dataset_root):
        # Try to load config from the log_dir first (config.yaml)
        config_path = os.path.join(log_dir, "config.yaml")
        in_t, out_t, stride = 1, 1, 1
        gt_dyn_vars = var_names  # fallback to prediction variable names

        if os.path.isfile(config_path):
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    train_cfg = yaml.safe_load(f)
                data_cfg = train_cfg.get("data", {})
                in_t = int(data_cfg.get("in_t", 1))
                out_t = int(data_cfg.get("out_t", 1))
                stride = int(data_cfg.get("stride", 1))
                cfg_dyn_vars = data_cfg.get("dyn_vars")
                if cfg_dyn_vars:
                    gt_dyn_vars = cfg_dyn_vars
            except Exception as e:
                result["warnings"].append(f"Failed to parse config.yaml: {e}")
        else:
            # Try to get config from dataset var_names.json
            ds_cfg = load_ground_truth_config(dataset_root)
            if ds_cfg["dyn_vars"]:
                gt_dyn_vars = ds_cfg["dyn_vars"]

        print(
            f"[Info] Loading ground truth: in_t={in_t}, out_t={out_t}, "
            f"stride={stride}, vars={gt_dyn_vars}",
            file=sys.stderr,
        )

        gt_data = load_ground_truth_test_set(
            dataset_root=dataset_root,
            dyn_vars=gt_dyn_vars,
            in_t=in_t,
            out_t=out_t,
            stride=stride,
        )

        if gt_data is not None:
            print(
                f"[Info] Ground truth loaded: shape={gt_data.shape} "
                f"(N_samples={gt_data.shape[0]}, out_t={gt_data.shape[1]}, "
                f"n_vars={gt_data.shape[2]}, H={gt_data.shape[3]}, W={gt_data.shape[4]})",
                file=sys.stderr,
            )
        else:
            result["warnings"].append(
                "Could not load ground truth from dataset_root. "
                "Plots will show prediction only (no comparison)."
            )
            print("[Warn] Ground truth not available, showing predictions only.", file=sys.stderr)
    else:
        if dataset_root is not None:
            result["warnings"].append(f"dataset_root does not exist: {dataset_root}")
        print("[Info] No dataset_root provided, showing predictions only.", file=sys.stderr)

    # ---- Generate per-sample comparison plots ----
    print(f"[Info] Generating per-sample plots for {len(vis_sample_ids)} samples...", file=sys.stderr)
    for si in vis_sample_ids:
        sample_data = pred_index.get(si, {})
        for ts_idx in sorted(sample_data.keys()):
            ts_data = sample_data[ts_idx]
            for var_idx in sorted(ts_data.keys()):
                var_info = ts_data[var_idx]
                var_name = var_info["varname"]

                # Load prediction
                try:
                    pred = np.load(var_info["path"]).astype(np.float32)
                except Exception as e:
                    result["warnings"].append(f"Failed to load {var_info['path']}: {e}")
                    continue

                # Load ground truth for this sample/timestep/var
                truth: Optional[np.ndarray] = None
                if gt_data is not None and si < gt_data.shape[0]:
                    if ts_idx < gt_data.shape[1] and var_idx < gt_data.shape[2]:
                        truth = gt_data[si, ts_idx, var_idx]
                        # Verify shape compatibility
                        if truth.shape != pred.shape:
                            result["warnings"].append(
                                f"Shape mismatch for sample {si} t{ts_idx} var{var_idx}: "
                                f"pred={pred.shape} vs truth={truth.shape}. Skipping truth."
                            )
                            truth = None

                out_name = f"predict_sample_{si}_var_{var_name}.png"
                if ts_idx > 0:
                    out_name = f"predict_sample_{si}_t{ts_idx}_var_{var_name}.png"
                out_path = os.path.join(output_dir, out_name)

                try:
                    plot_single_sample_var(
                        pred=pred,
                        truth=truth,
                        var_name=var_name,
                        sample_idx=si,
                        timestep=ts_idx,
                        output_path=out_path,
                    )
                    result["plots"].append(out_path)
                    print(f"  [{si}] {out_name}", file=sys.stderr)
                except Exception as e:
                    result["warnings"].append(f"Failed to generate {out_name}: {e}")

    # ---- Generate overview grid ----
    print("[Info] Generating overview grid...", file=sys.stderr)
    overview_path = os.path.join(output_dir, "predict_overview.png")
    try:
        plot_overview_grid(
            pred_index=pred_index,
            gt_data=gt_data,
            var_names=var_names,
            sample_indices=vis_sample_ids,
            output_path=overview_path,
        )
        result["plots"].append(overview_path)
        print(f"  predict_overview.png", file=sys.stderr)
    except Exception as e:
        result["warnings"].append(f"Failed to generate overview plot: {e}")

    # ---- Summary ----
    n_plots = len(result["plots"])
    result["n_samples"] = len(vis_sample_ids)
    result["n_total_samples"] = n_total
    result["n_variables"] = len(var_names)
    result["variable_names"] = var_names
    result["has_ground_truth"] = gt_data is not None

    if n_plots > 0:
        result["message"] = (
            f"Generated {n_plots} plots for {len(vis_sample_ids)} samples "
            f"({len(var_names)} variables) in {output_dir}"
        )
        print(f"[Success] {result['message']}", file=sys.stderr)
    else:
        result["status"] = "error"
        result["errors"].append("No plots were generated.")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prediction visualization plots for ocean forecast models."
    )
    parser.add_argument(
        "--log_dir",
        required=True,
        type=str,
        help="Training log directory (must contain predictions/ subdirectory)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Preprocessed dataset root (for ground truth comparison). Optional.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: {log_dir}/plots/)",
    )
    args = parser.parse_args()

    result = generate_predict_plots(
        log_dir=args.log_dir,
        dataset_root=args.dataset_root,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
    )

    # Output structured result for TypeScript process manager
    print(f"__result__{json.dumps(result, ensure_ascii=False)}__result__")


if __name__ == "__main__":
    main()
