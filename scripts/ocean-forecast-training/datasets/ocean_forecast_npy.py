"""
@file ocean_forecast_npy.py
@description Ocean forecast NPY dataset - loads Agent-preprocessed NPY time series data
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.0.0 initial version
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseDataset, TensorPairDataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class OceanForecastNpyDataset(BaseDataset):
    """
    Ocean Forecast NPY dataset.

    Loads time-series NPY files produced by the forecast data preprocessing
    pipeline (forecast_preprocess.py).  The preprocessing output layout is::

        dataset_root/
        ├── train/{var_name}/{date_str}.npy   # Each file: (H, W) float32
        ├── valid/{var_name}/{date_str}.npy
        ├── test/{var_name}/{date_str}.npy
        ├── static/{var_name}.npy             # Optional static variables
        ├── var_names.json                    # Variable metadata
        └── time_index.json                   # Sorted date lists per split

    Each ``__getitem__`` sample is a pair ``(x, y)`` where:

    - ``x``: input frames  – shape ``(H, W, in_t * C)``
    - ``y``: target frames – shape ``(H, W, out_t * C)``

    A sliding window of size ``in_t + out_t`` with step ``stride`` is used to
    generate samples from the temporal axis.

    Normalization is computed *per-channel* from the training split and applied
    consistently to all splits.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, data_args: Dict[str, Any], **kwargs: Any) -> None:
        # ---- hyper-parameters from YAML config ----
        self.data_path: str = data_args.get("data_path", "")
        self.in_t: int = int(data_args.get("in_t", 1))
        self.out_t: int = int(data_args.get("out_t", 1))
        self.stride: int = int(data_args.get("stride", 1))
        self.normalize: bool = bool(data_args.get("normalize", True))
        self.normalizer_type: str = str(data_args.get("normalizer_type", "PGN"))
        self.reprocess: bool = bool(data_args.get("reprocess", False))

        # Store the full data_args so BaseDataset helpers (make_loaders) work.
        self.data_args = data_args

        # ---- read metadata from preprocessing outputs ----
        self._dataset_root: str = self.data_path
        var_names_path = osp.join(self._dataset_root, "var_names.json")
        time_index_path = osp.join(self._dataset_root, "time_index.json")

        if not osp.isfile(var_names_path):
            raise FileNotFoundError(
                f"var_names.json not found at {var_names_path}. "
                "Please run the forecast preprocessing pipeline first."
            )
        if not osp.isfile(time_index_path):
            raise FileNotFoundError(
                f"time_index.json not found at {time_index_path}. "
                "Please run the forecast preprocessing pipeline first."
            )

        with open(var_names_path, "r", encoding="utf-8") as f:
            var_names_cfg = json.load(f)

        with open(time_index_path, "r", encoding="utf-8") as f:
            time_index_cfg = json.load(f)

        # Dynamic variable names – allow override from YAML config.
        self.dyn_vars: List[str] = data_args.get(
            "dyn_vars",
            var_names_cfg.get("dynamic", var_names_cfg.get("dyn_vars", [])),
        )
        if not self.dyn_vars:
            raise ValueError(
                "No dynamic variables found. Check var_names.json or "
                "provide 'dyn_vars' in the YAML config."
            )

        # Spatial shape from preprocessing metadata.
        spatial_shape_raw = var_names_cfg.get("spatial_shape")
        if spatial_shape_raw is None or len(spatial_shape_raw) < 2:
            raise ValueError(
                "spatial_shape not found or invalid in var_names.json."
            )
        self.spatial_shape: Tuple[int, int] = (
            int(spatial_shape_raw[-2]),
            int(spatial_shape_raw[-1]),
        )

        # Per-split filename lists (NPY file stems, without extension).
        self._split_filenames: Dict[str, List[str]] = {}
        for split in ("train", "valid", "test"):
            split_info = time_index_cfg.get("splits", {}).get(split, {})
            filenames = split_info.get("filenames")
            if filenames is None:
                # Fallback: use timestamps truncated to the same length as
                # the first available filename, or the raw timestamps.
                filenames = split_info.get("timestamps", [])
            self._split_filenames[split] = filenames

        # Validate that we have at least the training split.
        if not self._split_filenames["train"]:
            raise ValueError(
                "Training split has zero time steps in time_index.json."
            )

        # ---- derived constants ----
        n_vars = len(self.dyn_vars)
        self._window_size: int = self.in_t + self.out_t

        # ---- normalizers (populated during processing) ----
        self.x_normalizer: Optional[Any] = None
        self.y_normalizer: Optional[Any] = None

        # ---- geometry / metadata (exposed for the trainer) ----
        H, W = self.spatial_shape
        self.geom: Optional[Dict[str, Any]] = {
            "dim": 2,
            "layout": "grid",
            "spatial_shape": self.spatial_shape,
            "in_channels": self.in_t * n_vars,
            "out_channels": self.out_t * n_vars,
            "dyn_vars": list(self.dyn_vars),
            "in_t": self.in_t,
            "out_t": self.out_t,
        }
        self.coords: Optional[torch.Tensor] = None

        # ---- cache path ----
        self.cache_path: str = self._build_cache_path()

        # ---- load or process all splits ----
        (
            train_x,
            train_y,
            valid_x,
            valid_y,
            test_x,
            test_y,
            self.x_normalizer,
            self.y_normalizer,
        ) = self._load_or_process_splits()

        # ---- optional subset (quick experiments) ----
        self.subset: bool = data_args.get("subset", False)
        self.subset_ratio: float = data_args.get("subset_ratio", 0.1)
        if self.subset:

            def _apply_subset(
                x: torch.Tensor, y: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                n = max(1, int(len(x) * self.subset_ratio))
                return x[:n], y[:n]

            train_x, train_y = _apply_subset(train_x, train_y)
            valid_x, valid_y = _apply_subset(valid_x, valid_y)
            test_x, test_y = _apply_subset(test_x, test_y)

        # ---- wrap into TensorPairDataset ----
        self.train_dataset = TensorPairDataset(train_x, train_y, mode="train")
        self.valid_dataset = TensorPairDataset(valid_x, valid_y, mode="valid")
        self.test_dataset = TensorPairDataset(test_x, test_y, mode="test")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _build_cache_path(self) -> str:
        """Build a deterministic cache filename encoding all relevant params."""
        norm_tag = self.normalizer_type if self.normalize else "none"
        # Include a hash of dyn_vars to avoid collisions when var subsets change.
        vars_hash = hashlib.md5(
            ",".join(sorted(self.dyn_vars)).encode()
        ).hexdigest()[:8]
        cache_name = (
            f"ocean_forecast_npy"
            f"_it{self.in_t}_ot{self.out_t}_s{self.stride}"
            f"_norm{norm_tag}_v{vars_hash}_processed.pt"
        )
        return osp.join(self._dataset_root, cache_name)

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------
    def _load_split_tensor(self, split: str) -> torch.Tensor:
        """Load all NPY files for *split* and stack as ``(T, H, W, C)`` tensor.

        Each NPY file is ``(H, W)`` float32 for a single variable at a single
        time step.  We iterate over filenames (sorted temporally by the
        preprocessing pipeline) and variables to assemble a 4-D tensor.

        Returns:
            Tensor of shape ``(T, H, W, C)`` where ``C = len(dyn_vars)``.
        """
        filenames = self._split_filenames[split]
        if not filenames:
            raise ValueError(f"Split '{split}' has no time steps.")

        H, W = self.spatial_shape
        T = len(filenames)
        C = len(self.dyn_vars)

        # Pre-allocate; fill with zeros (NaN → 0 will happen later, but this
        # also handles any missing-file edge case gracefully).
        tensor = torch.zeros(T, H, W, C, dtype=torch.float32)

        for t_idx, fname in enumerate(filenames):
            for c_idx, var_name in enumerate(self.dyn_vars):
                npy_path = osp.join(
                    self._dataset_root, split, var_name, f"{fname}.npy"
                )
                if not osp.isfile(npy_path):
                    raise FileNotFoundError(
                        f"Missing NPY file: {npy_path}. "
                        f"The preprocessing output may be incomplete."
                    )
                arr = np.load(npy_path).astype(np.float32)

                # Handle spatial shape: the preprocessing may produce (H, W)
                # or (D, H, W) for depth-resolved variables.  We take the last
                # two spatial dims.
                if arr.ndim == 2:
                    if arr.shape != (H, W):
                        raise ValueError(
                            f"Shape mismatch for {npy_path}: "
                            f"expected ({H}, {W}), got {arr.shape}."
                        )
                    frame = torch.from_numpy(arr)
                elif arr.ndim == 3:
                    # Depth-resolved: take the first depth level.
                    frame = torch.from_numpy(arr[0])
                    if frame.shape != (H, W):
                        raise ValueError(
                            f"Shape mismatch for {npy_path} (after depth "
                            f"slice): expected ({H}, {W}), got {frame.shape}."
                        )
                else:
                    raise ValueError(
                        f"Unexpected ndim={arr.ndim} for {npy_path}."
                    )

                tensor[t_idx, :, :, c_idx] = frame

        # Replace NaN with 0.
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            tensor = torch.where(nan_mask, torch.zeros_like(tensor), tensor)

        return tensor

    def _create_sliding_windows(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window to produce (x, y) sample pairs.

        Args:
            data: Tensor of shape ``(T, H, W, C)``.

        Returns:
            x: ``(N_samples, H, W, in_t * C)``
            y: ``(N_samples, H, W, out_t * C)``
        """
        T, H, W, C = data.shape

        if T < self._window_size:
            raise ValueError(
                f"Not enough time steps for sliding window: T={T}, "
                f"window_size={self._window_size} (in_t={self.in_t} + "
                f"out_t={self.out_t})."
            )

        starts = list(range(0, T - self._window_size + 1, self.stride))
        if not starts:
            raise ValueError(
                f"Stride {self.stride} too large for T={T}, "
                f"window_size={self._window_size}."
            )

        N = len(starts)
        x = torch.empty(N, H, W, self.in_t * C, dtype=data.dtype)
        y = torch.empty(N, H, W, self.out_t * C, dtype=data.dtype)

        for i, t0 in enumerate(starts):
            # Input frames: [t0, t0 + in_t)
            x_frames = data[t0 : t0 + self.in_t]  # (in_t, H, W, C)
            # Target frames: [t0 + in_t, t0 + in_t + out_t)
            y_frames = data[t0 + self.in_t : t0 + self.in_t + self.out_t]  # (out_t, H, W, C)

            # Reshape: (in_t, H, W, C) → (H, W, in_t*C)
            x[i] = x_frames.permute(1, 2, 0, 3).reshape(H, W, self.in_t * C)
            y[i] = y_frames.permute(1, 2, 0, 3).reshape(H, W, self.out_t * C)

        return x, y

    def _compute_normalizer(
        self, data: torch.Tensor
    ) -> Optional[Any]:
        """Compute a normalizer from training data.

        Args:
            data: Tensor of shape ``(N, H, W, C)`` (channel-last).

        Returns:
            A normalizer instance (UnitGaussianNormalizer or GaussianNormalizer),
            or ``None`` if normalization is disabled.
        """
        if not self.normalize:
            return None

        if self.normalizer_type == "PGN":
            return UnitGaussianNormalizer(data)
        elif self.normalizer_type == "GN":
            return GaussianNormalizer(data)
        else:
            raise ValueError(
                f"Unknown normalizer_type: '{self.normalizer_type}'. "
                f"Supported values: 'PGN' (UnitGaussianNormalizer), "
                f"'GN' (GaussianNormalizer)."
            )

    def _load_or_process_splits(self) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Any],
        Optional[Any],
    ]:
        """Load from cache or process all three splits from scratch.

        Returns:
            (train_x, train_y, valid_x, valid_y, test_x, test_y,
             x_normalizer, y_normalizer)
        """
        # ---- try cache ----
        if osp.isfile(self.cache_path) and not self.reprocess:
            print(f"[OceanForecastNpy] Loading from cache: {self.cache_path}")
            return self._load_from_cache()

        print("[OceanForecastNpy] Processing NPY data from scratch ...")

        # ---- load raw tensors per split ----
        print("  Loading train split ...")
        train_raw = self._load_split_tensor("train")
        print(f"    train raw shape: {tuple(train_raw.shape)}")

        print("  Loading valid split ...")
        valid_raw = self._load_split_tensor("valid")
        print(f"    valid raw shape: {tuple(valid_raw.shape)}")

        print("  Loading test split ...")
        test_raw = self._load_split_tensor("test")
        print(f"    test raw shape: {tuple(test_raw.shape)}")

        # ---- sliding window ----
        print("  Creating sliding windows ...")
        train_x, train_y = self._create_sliding_windows(train_raw)
        valid_x, valid_y = self._create_sliding_windows(valid_raw)
        test_x, test_y = self._create_sliding_windows(test_raw)

        print(
            f"    samples: train={len(train_x)}, "
            f"valid={len(valid_x)}, test={len(test_x)}"
        )

        # ---- normalization ----
        x_normalizer: Optional[Any] = None
        y_normalizer: Optional[Any] = None

        if self.normalize:
            print(
                f"  Computing normalizer (type={self.normalizer_type}) "
                f"from training data ..."
            )
            x_normalizer = self._compute_normalizer(train_x)
            y_normalizer = self._compute_normalizer(train_y)

            if x_normalizer is not None:
                train_x = x_normalizer.encode(train_x)
                valid_x = x_normalizer.encode(valid_x)
                test_x = x_normalizer.encode(test_x)

            if y_normalizer is not None:
                train_y = y_normalizer.encode(train_y)
                valid_y = y_normalizer.encode(valid_y)
                test_y = y_normalizer.encode(test_y)

            print("    Normalization applied to all splits.")

        # ---- save to cache ----
        self._save_to_cache(
            train_x, train_y,
            valid_x, valid_y,
            test_x, test_y,
            x_normalizer, y_normalizer,
        )

        return (
            train_x, train_y,
            valid_x, valid_y,
            test_x, test_y,
            x_normalizer, y_normalizer,
        )

    def _save_to_cache(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        valid_x: torch.Tensor,
        valid_y: torch.Tensor,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
        x_normalizer: Optional[Any],
        y_normalizer: Optional[Any],
    ) -> None:
        """Persist processed tensors and normalizers to a ``.pt`` file."""
        cache_dir = osp.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        payload = {
            "train_x": train_x,
            "train_y": train_y,
            "valid_x": valid_x,
            "valid_y": valid_y,
            "test_x": test_x,
            "test_y": test_y,
            "x_normalizer": x_normalizer,
            "y_normalizer": y_normalizer,
            "geom": self.geom,
            "coords": self.coords,
        }
        torch.save(payload, self.cache_path)
        print(f"  Cache saved to: {self.cache_path}")

    def _load_from_cache(self) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Any],
        Optional[Any],
    ]:
        """Load processed tensors and normalizers from a cached ``.pt`` file."""
        obj = torch.load(self.cache_path, weights_only=False)

        self.geom = obj.get("geom", self.geom)
        self.coords = obj.get("coords", None)

        return (
            obj["train_x"],
            obj["train_y"],
            obj["valid_x"],
            obj["valid_y"],
            obj["test_x"],
            obj["test_y"],
            obj.get("x_normalizer"),
            obj.get("y_normalizer"),
        )

    # ------------------------------------------------------------------
    # BaseDataset hook overrides (not used – we bypass load_or_process)
    # ------------------------------------------------------------------
    def get_cache_path(self) -> str:
        """Overridden but not called directly; see ``_build_cache_path``."""
        return getattr(self, "cache_path", "ocean_forecast_npy_processed.pt")

    def load_raw_data(self, **kwargs: Any) -> Any:
        """Not used – data loading is handled in ``_load_or_process_splits``."""
        raise NotImplementedError(
            "OceanForecastNpyDataset does not use load_raw_data(). "
            "Data is loaded per-split via _load_split_tensor()."
        )

    def process_split(
        self,
        data_split: Any,
        mode: str,
        x_normalizer: Optional[Any] = None,
        y_normalizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any], Optional[Any]]:
        """Not used – split processing is handled in ``_load_or_process_splits``."""
        raise NotImplementedError(
            "OceanForecastNpyDataset does not use process_split(). "
            "Processing is done in _load_or_process_splits()."
        )
