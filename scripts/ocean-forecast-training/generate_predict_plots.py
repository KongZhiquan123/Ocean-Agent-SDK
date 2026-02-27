#!/usr/bin/env python3
"""
@file generate_predict_plots.py
@description Generate prediction visualization plots for ocean forecast models.
@author Leizheng
@date 2026-02-26
@version 1.3.0

@changelog
  - 2026-02-27 Leizheng: v1.3.0 add date to titles from time_index.json, reduce title-plot
        spacing, add error column to overview grid
  - 2026-02-27 Leizheng: v1.2.0 white bg + tan land mask, equal aspect, default n_samples=3
  - 2026-02-26 Leizheng: v1.1.0 add coordinate axes, land mask, YlOrRd error cmap
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
except ImportError:
    DEFAULT_CMAP = "viridis"

# Error colormap: YlOrRd gives white→yellow→orange→red,
# low-error regions are white which clearly separates from gray land mask.
ERROR_CMAP = "YlOrRd"


# ============================================================================
# Coordinate / mask helpers
# ============================================================================

def _make_grid_edges(centers: np.ndarray) -> np.ndarray:
    """Convert N centre-point coordinates to N+1 edge coordinates for pcolormesh(shading='flat')."""
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def load_static_data(dataset_root: Optional[str]) -> Dict[str, Optional[np.ndarray]]:
    """Load lon, lat, and mask arrays from the preprocessed dataset.

    Reads ``var_names.json`` for ``lon_var``, ``lat_var``, ``mask`` field names,
    then loads the corresponding ``.npy`` files from ``static_variables/``.
    File names carry a numeric prefix (e.g. ``40_lon_rho.npy``); the variable
    name is extracted via ``base.split('_', 1)[-1]``.

    Returns:
        ``{'lon': ndarray|None, 'lat': ndarray|None, 'mask': ndarray|None}``
    """
    result: Dict[str, Optional[np.ndarray]] = {"lon": None, "lat": None, "mask": None}
    if dataset_root is None:
        return result

    # Read var_names.json
    var_names_path = os.path.join(dataset_root, "var_names.json")
    lon_key: Optional[str] = None
    lat_key: Optional[str] = None
    mask_key: Optional[str] = None

    if os.path.isfile(var_names_path):
        try:
            with open(var_names_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            lon_key = cfg.get("lon_var")
            lat_key = cfg.get("lat_var")
            # mask can be a list of mask var names (e.g. ["mask_rho","mask_u",...])
            # or a single string; we only need the primary mask (first element).
            mask_val = cfg.get("mask")
            if isinstance(mask_val, list):
                mask_key = mask_val[0] if mask_val else None
            else:
                mask_key = mask_val
        except Exception:
            pass

    # Scan static_variables/ directory
    static_dir = os.path.join(dataset_root, "static_variables")
    if not os.path.isdir(static_dir):
        return result

    # Build name→path mapping from numbered-prefix filenames
    name_to_path: Dict[str, str] = {}
    for fname in os.listdir(static_dir):
        if not fname.endswith(".npy"):
            continue
        base = fname[:-4]  # strip .npy
        var_name = base.split("_", 1)[-1] if "_" in base else base
        name_to_path[var_name] = os.path.join(static_dir, fname)

    # Load by explicit key or fallback keyword matching
    def _load(key: Optional[str], keywords: List[str]) -> Optional[np.ndarray]:
        if key and key in name_to_path:
            return np.load(name_to_path[key])
        # Fallback: match by keyword
        for k in keywords:
            for name, fpath in name_to_path.items():
                if k in name.lower():
                    return np.load(fpath)
        return None

    result["lon"] = _load(lon_key, ["lon"])
    result["lat"] = _load(lat_key, ["lat"])
    result["mask"] = _load(mask_key, ["mask"])

    return result


# ============================================================================
# Prediction file parsing
# ============================================================================

# Expected filename pattern produced by trainers/base.py predict():
#   sample_{i:06d}_t{t}_var{c}_{var_name}.npy
_PRED_PATTERN = re.compile(
    r"^sample_(?P<sample>\d+)_t(?P<timestep>\d+)_var(?P<varidx>\d+)_(?P<varname>.+)\.npy$"
)

# Ground truth files saved alongside predictions by predict()
_TRUTH_PATTERN = re.compile(
    r"^truth_(?P<sample>\d+)_t(?P<timestep>\d+)_var(?P<varidx>\d+)_(?P<varname>.+)\.npy$"
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
# Ground truth loading (from truth_*.npy files saved by predict())
# ============================================================================

def load_ground_truth_from_pred_dir(
    pred_dir: str,
) -> Optional[np.ndarray]:
    """Load ground truth from truth_*.npy files saved alongside predictions.

    The predict() method in trainers/base.py saves denormalized ground truth
    files with the naming convention: truth_{sample}_t{t}_var{c}_{varname}.npy

    Returns:
        Array of shape (N_samples, out_t, n_vars, H, W), or None if no
        truth files are found.
    """
    if not os.path.isdir(pred_dir):
        return None

    truth_files: Dict[Tuple[int, int, int], str] = {}
    max_sample = -1
    max_ts = -1
    max_var = -1

    for fname in os.listdir(pred_dir):
        m = _TRUTH_PATTERN.match(fname)
        if m is None:
            continue
        s = int(m.group("sample"))
        t = int(m.group("timestep"))
        v = int(m.group("varidx"))
        truth_files[(s, t, v)] = os.path.join(pred_dir, fname)
        max_sample = max(max_sample, s)
        max_ts = max(max_ts, t)
        max_var = max(max_var, v)

    if not truth_files:
        return None

    # Determine spatial shape from the first file
    first_path = next(iter(truth_files.values()))
    first_arr = np.load(first_path)
    spatial = first_arr.shape  # (H, W)

    N = max_sample + 1
    out_t = max_ts + 1
    C = max_var + 1

    gt_data = np.zeros((N, out_t, C, *spatial), dtype=np.float32)
    for (s, t, v), fpath in truth_files.items():
        gt_data[s, t, v] = np.load(fpath).astype(np.float32)

    return gt_data


# ============================================================================
# Ground truth loading (from dataset_root)

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


def load_time_index(
    dataset_root: Optional[str],
    in_t: int = 7,
    stride: int = 1,
) -> Dict[int, str]:
    """Build a mapping from sample_idx → date string using time_index.json.

    The sliding window offset is: target_time = test_timestamps[in_t + sample_idx * stride].
    Returns {sample_idx: "YYYY-MM-DD"} for all computable indices.
    """
    if dataset_root is None:
        return {}

    ti_path = os.path.join(dataset_root, "time_index.json")
    if not os.path.isfile(ti_path):
        return {}

    try:
        with open(ti_path, "r", encoding="utf-8") as f:
            ti = json.load(f)
        test_info = ti.get("splits", {}).get("test", {})
        timestamps: List[str] = test_info.get("timestamps", [])
        if not timestamps:
            return {}

        result: Dict[int, str] = {}
        total_test = len(timestamps)
        window_size = in_t + 1  # out_t is always at least 1
        n_samples = (total_test - window_size) // stride + 1

        for si in range(n_samples):
            target_idx = in_t + si * stride
            if target_idx < total_test:
                ts = timestamps[target_idx]  # e.g. "19940707120000"
                # Parse to "YYYY-MM-DD"
                if len(ts) >= 8:
                    result[si] = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        return result
    except Exception:
        return {}


# ============================================================================
# Plotting functions
# ============================================================================

def _add_colorbar(fig: plt.Figure, ax: plt.Axes, im: Any) -> None:
    """Add a thin colorbar next to an axis without changing its size."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.06)
    fig.colorbar(im, cax=cax)


def _plot_panel(
    ax: plt.Axes,
    data: np.ndarray,
    lon: Optional[np.ndarray],
    lat: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    cmap,
    vmin: float,
    vmax: float,
) -> Any:
    """Unified panel drawing with optional coordinate axes and land masking.

    * Applies land mask (mask==0 → NaN, shown as light-gray via ``set_bad``).
    * If *lon*/*lat* are provided, uses ``pcolormesh`` with proper edges;
      otherwise falls back to pixel-index ``pcolormesh``.
    * Returns the QuadMesh (or equivalent) for colorbar attachment.
    """
    display = data.astype(np.float64, copy=True)

    # ---- Apply land mask ----
    if mask is not None:
        if mask.shape == display.shape:
            display[mask == 0] = np.nan
        else:
            # Nearest-neighbour resample mask to data shape
            try:
                from scipy.ndimage import zoom
                factors = (display.shape[0] / mask.shape[0],
                           display.shape[1] / mask.shape[1])
                resampled = zoom(mask.astype(np.float32), factors, order=0)
                display[resampled < 0.5] = np.nan
            except ImportError:
                pass  # scipy unavailable, skip mask resampling

    # ---- Prepare colormap with land color ----
    # Land = NaN → shown as light tan (cartographic convention for land)
    # Axes background = white (clearly distinct from land)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()
    else:
        cmap = cmap.copy()
    cmap.set_bad(color="#e8dcc8")  # light tan for land

    # ---- Validate coordinate dimensions against data ----
    use_coords = False
    if lon is not None and lat is not None:
        H, W = display.shape
        if lon.ndim == 1 and lat.ndim == 1:
            use_coords = (lon.size == W and lat.size == H)
        elif lon.ndim == 2 and lat.ndim == 2:
            use_coords = (lon.shape == display.shape and lat.shape == display.shape)

    # ---- White background so it's distinct from the gray land mask ----
    ax.set_facecolor("white")

    # ---- Draw ----
    if use_coords:
        if lon.ndim == 1 and lat.ndim == 1:
            # 1-D rectilinear → build edge grids
            lon_e = _make_grid_edges(lon)
            lat_e = _make_grid_edges(lat)
            im = ax.pcolormesh(
                lon_e, lat_e, display,
                cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
            )
        else:
            # 2-D curvilinear → crop axes to data extent
            im = ax.pcolormesh(
                lon, lat, display,
                cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
            )
            # Tight limits: remove the empty white margin outside the grid
            ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
            ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
            ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    else:
        # No valid coordinates → pixel indices
        im = ax.pcolormesh(
            display,
            cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    return im


def plot_single_sample_var(
    pred: np.ndarray,
    truth: Optional[np.ndarray],
    var_name: str,
    sample_idx: int,
    timestep: int,
    output_path: str,
    dpi: int = 150,
    lon: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    date_str: Optional[str] = None,
) -> str:
    """Plot a 3-panel (or 1-panel) comparison for one sample/variable/timestep.

    Panels: Prediction | Ground Truth | Absolute Error
    If ground truth is unavailable, only the prediction panel is shown.
    Supports optional lon/lat coordinate axes and land mask overlay.

    Returns:
        The output_path on success.
    """
    has_truth = truth is not None

    if has_truth:
        n_cols = 3
        cell = 5.0
        fig, axes = plt.subplots(1, n_cols, figsize=(cell * n_cols + 0.5, cell), dpi=dpi)

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
        im0 = _plot_panel(axes[0], pred, lon, lat, mask, DEFAULT_CMAP, vmin, vmax)
        axes[0].set_title("Prediction", fontsize=11, fontweight="bold")
        _add_colorbar(fig, axes[0], im0)

        # Panel 2: Ground Truth
        im1 = _plot_panel(axes[1], truth, lon, lat, mask, DEFAULT_CMAP, vmin, vmax)
        axes[1].set_title("Ground Truth", fontsize=11, fontweight="bold")
        _add_colorbar(fig, axes[1], im1)

        # Panel 3: Absolute Error
        im2 = _plot_panel(axes[2], error, lon, lat, mask, ERROR_CMAP, 0, emax)
        axes[2].set_title("|Pred - Truth|", fontsize=11, fontweight="bold")
        _add_colorbar(fig, axes[2], im2)

    else:
        # No ground truth: single panel
        n_cols = 1
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
        axes = [ax]

        finite = pred[np.isfinite(pred)]
        if finite.size > 0:
            vmin = float(np.nanpercentile(finite, 1))
            vmax = float(np.nanpercentile(finite, 99))
        else:
            vmin, vmax = 0.0, 1.0

        im0 = _plot_panel(ax, pred, lon, lat, mask, DEFAULT_CMAP, vmin, vmax)
        ax.set_title("Prediction", fontsize=11, fontweight="bold")
        _add_colorbar(fig, ax, im0)

    title_parts = [f"Sample {sample_idx}"]
    if date_str:
        title_parts.append(date_str)
    if timestep > 0:
        title_parts.append(f"t={timestep}")
    title_parts.append(var_name)
    fig.suptitle(
        "  |  ".join(title_parts),
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def plot_overview_grid(
    pred_index: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]],
    gt_data: Optional[np.ndarray],
    var_names: List[str],
    sample_indices: List[int],
    output_path: str,
    dpi: int = 120,
    lon: Optional[np.ndarray] = None,
    lat: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    date_map: Optional[Dict[int, str]] = None,
) -> str:
    """Generate an overview grid showing all visualized samples.

    Layout per row (one sample):
      - For each variable: Prediction | |Pred - Truth| (if GT available)
      - Without GT: just Prediction per variable.

    Returns:
        The output_path on success.
    """
    n_rows = len(sample_indices)
    n_vars = len(var_names)
    has_gt = gt_data is not None

    if n_rows == 0 or n_vars == 0:
        return output_path

    # Columns: per variable show pred + error (if GT), else just pred
    if has_gt:
        n_cols = n_vars * 2  # pred, error for each variable
    else:
        n_cols = n_vars

    cell_w = 4.5
    cell_h = 4.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols + 1.5, cell_h * n_rows + 1.0),
        dpi=dpi,
        squeeze=False,
    )

    for row, si in enumerate(sample_indices):
        sample_data = pred_index.get(si, {})
        ts_data = sample_data.get(0, {})

        for vi, var_name in enumerate(var_names):
            var_info = ts_data.get(vi)

            if has_gt:
                ax_pred = axes[row, vi * 2]
                ax_err = axes[row, vi * 2 + 1]
            else:
                ax_pred = axes[row, vi]
                ax_err = None

            if var_info is None:
                ax_pred.text(0.5, 0.5, "N/A", ha="center", va="center",
                             transform=ax_pred.transAxes, fontsize=12, color="gray")
                ax_pred.set_xticks([])
                ax_pred.set_yticks([])
                if ax_err is not None:
                    ax_err.text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=ax_err.transAxes, fontsize=12, color="gray")
                    ax_err.set_xticks([])
                    ax_err.set_yticks([])
            else:
                pred = np.load(var_info["path"]).astype(np.float32)
                finite = pred[np.isfinite(pred)]
                if finite.size > 0:
                    vmin = float(np.nanpercentile(finite, 1))
                    vmax = float(np.nanpercentile(finite, 99))
                else:
                    vmin, vmax = 0.0, 1.0

                im = _plot_panel(ax_pred, pred, lon, lat, mask, DEFAULT_CMAP, vmin, vmax)
                _add_colorbar(fig, ax_pred, im)

                # Hide x ticks for non-bottom rows
                if row < n_rows - 1:
                    ax_pred.tick_params(axis='x', labelbottom=False, length=0)
                # Hide y ticks for non-first pred columns
                if vi > 0:
                    ax_pred.tick_params(axis='y', labelleft=False, length=0)

                # Error panel
                if ax_err is not None and has_gt and si < gt_data.shape[0]:
                    truth = gt_data[si, 0, vi]
                    error = np.abs(pred - truth)
                    emax = float(np.nanpercentile(
                        error[np.isfinite(error)], 99)) if np.any(np.isfinite(error)) else 1.0
                    im_e = _plot_panel(ax_err, error, lon, lat, mask, ERROR_CMAP, 0, emax)
                    _add_colorbar(fig, ax_err, im_e)

                    rmse = float(np.sqrt(np.nanmean((pred - truth) ** 2)))
                    mae = float(np.nanmean(np.abs(pred - truth)))
                    ax_err.text(
                        0.02, 0.96,
                        f"RMSE={rmse:.4g}\nMAE={mae:.4g}",
                        transform=ax_err.transAxes, fontsize=7,
                        color="white", verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
                    )
                    if row < n_rows - 1:
                        ax_err.tick_params(axis='x', labelbottom=False, length=0)
                    ax_err.tick_params(axis='y', labelleft=False, length=0)
                elif ax_err is not None:
                    ax_err.set_visible(False)

            # Column headers (first row)
            if row == 0:
                ax_pred.set_title(f"{var_name}", fontsize=10, fontweight="bold")
                if ax_err is not None:
                    ax_err.set_title(f"|Error|", fontsize=10, fontweight="bold")

            # Row labels (first column)
            if (has_gt and vi == 0) or (not has_gt and vi == 0):
                date_label = f" ({date_map[si]})" if date_map and si in date_map else ""
                ax_pred.set_ylabel(f"Sample {si}{date_label}", fontsize=9, fontweight="medium")

    fig.suptitle(
        "Prediction Overview",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(h_pad=0.3, w_pad=1.0)
    fig.subplots_adjust(top=0.95)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
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

    # ---- Load ground truth ----
    gt_data: Optional[np.ndarray] = None

    # Priority 1: Truth files saved alongside predictions by predict()
    gt_data = load_ground_truth_from_pred_dir(pred_dir)
    if gt_data is not None:
        print(
            f"[Info] Ground truth loaded from prediction dir: shape={gt_data.shape} "
            f"(N={gt_data.shape[0]}, out_t={gt_data.shape[1]}, "
            f"n_vars={gt_data.shape[2]}, H={gt_data.shape[3]}, W={gt_data.shape[4]})",
            file=sys.stderr,
        )

    # Priority 2: Load from dataset_root (fallback)
    if gt_data is None:
        # Auto-detect dataset_root from config.yaml if not provided
        if dataset_root is None:
            config_path = os.path.join(log_dir, "config.yaml")
            if os.path.isfile(config_path):
                try:
                    import yaml
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    data_cfg_auto = cfg.get("data", {})
                    dataset_root = data_cfg_auto.get("dataset_root") or data_cfg_auto.get("data_path")
                    if dataset_root:
                        print(f"[Info] Auto-detected dataset_root from config.yaml: {dataset_root}", file=sys.stderr)
                except Exception:
                    pass

            # Also try predict_meta.json
            if dataset_root is None:
                meta_path = os.path.join(pred_dir, "predict_meta.json")
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        dataset_root = meta.get("dataset_root")
                        if dataset_root:
                            print(f"[Info] Auto-detected dataset_root from predict_meta.json: {dataset_root}", file=sys.stderr)
                    except Exception:
                        pass

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
                f"[Info] Loading ground truth from dataset_root: in_t={in_t}, out_t={out_t}, "
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
                    f"(N={gt_data.shape[0]}, out_t={gt_data.shape[1]}, "
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
            print("[Info] No ground truth available, showing predictions only.", file=sys.stderr)

    # ---- Load static data (coordinates + land mask) ----
    # Resolve dataset_root for static data loading (auto-detect from predict_meta if needed)
    effective_dataset_root = dataset_root
    if effective_dataset_root is None:
        _meta_path = os.path.join(pred_dir, "predict_meta.json")
        if os.path.isfile(_meta_path):
            try:
                with open(_meta_path, "r", encoding="utf-8") as f:
                    _pmeta = json.load(f)
                effective_dataset_root = _pmeta.get("dataset_root")
            except Exception:
                pass
    static_data = load_static_data(effective_dataset_root)
    s_lon = static_data["lon"]
    s_lat = static_data["lat"]
    s_mask = static_data["mask"]
    if s_lon is not None or s_lat is not None:
        print(
            f"[Info] Static data loaded: lon={'(' + 'x'.join(str(x) for x in s_lon.shape) + ')' if s_lon is not None else 'None'}, "
            f"lat={'(' + 'x'.join(str(x) for x in s_lat.shape) + ')' if s_lat is not None else 'None'}, "
            f"mask={'(' + 'x'.join(str(x) for x in s_mask.shape) + ')' if s_mask is not None else 'None'}",
            file=sys.stderr,
        )

    # ---- Load time index for date labels ----
    # Determine in_t/stride for time_index mapping
    ti_in_t = 7
    ti_stride = 1
    # Auto-detect dataset_root if not yet set (for time_index loading)
    ti_dataset_root = dataset_root
    meta_path = os.path.join(pred_dir, "predict_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                pmeta = json.load(f)
            ti_in_t = int(pmeta.get("in_t", ti_in_t))
            ti_stride = int(pmeta.get("stride", ti_stride))
            if ti_dataset_root is None:
                ti_dataset_root = pmeta.get("dataset_root")
        except Exception:
            pass
    date_map = load_time_index(ti_dataset_root, in_t=ti_in_t, stride=ti_stride)
    if date_map:
        print(f"[Info] Time index loaded: {len(date_map)} sample-to-date mappings", file=sys.stderr)

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
                        lon=s_lon,
                        lat=s_lat,
                        mask=s_mask,
                        date_str=date_map.get(si),
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
            lon=s_lon,
            lat=s_lat,
            mask=s_mask,
            date_map=date_map,
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
        default=3,
        help="Number of samples to visualize (default: 3)",
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
