"""
@file forecast_adapter.py

@description Adapter that bridges NeuralFramework models (B,T,C,H,W) to SDK format (B,H,W,T*C).
@author Leizheng
@date 2026-02-27
@version 1.0.0

@changelog
  - 2026-02-27 Leizheng: v1.0.0 initial creation
"""

import torch
import torch.nn as nn


class ForecastModelAdapter(nn.Module):
    """Adapt NeuralFramework models to the SDK tensor convention.

    SDK convention:  input  (B, H, W, in_t * C)   →  output (B, H, W, out_t * C)
    NF  convention:  input  (B, T_in, C, H, W)    →  output (B, T_out, C, H, W)

    The adapter:
      1. Recovers (B, H, W, in_t*C) → (B, in_t, num_vars, H, W)
      2. Calls the inner NF model
      3. Converts output back to (B, H, W, out_t*C)

    ``model_params`` must contain ``in_channels`` (= in_t * num_vars) so that
    ``num_vars`` can be derived.  ``in_t`` and ``out_t`` default to 7 and 1.
    """

    def __init__(self, model_params: dict, model_cls: type, **kwargs):
        super().__init__()

        self.in_t = model_params.get('in_t', 7)
        self.out_t = model_params.get('out_t', 1)

        # SDK passes in_channels = in_t * num_vars
        total_in = model_params['in_channels']
        if total_in % self.in_t != 0:
            raise ValueError(
                f"in_channels ({total_in}) must be divisible by in_t ({self.in_t})"
            )
        self.num_vars = total_in // self.in_t

        # Build the inner NF model with adjusted parameters
        inner_params = dict(model_params)
        inner_params['input_len'] = self.in_t
        inner_params['output_len'] = self.out_t
        inner_params['in_channels'] = self.num_vars

        self.inner_model = model_cls(inner_params)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, in_t * num_vars)  – SDK convention
        Returns:
            (B, H, W, out_t * num_vars)    – SDK convention
        """
        B, H, W, _ = x.shape

        # (B, H, W, in_t*C) → (B, H, W, in_t, C) → (B, in_t, C, H, W)
        x = x.reshape(B, H, W, self.in_t, self.num_vars)
        x = x.permute(0, 3, 4, 1, 2)  # (B, T, C, H, W)

        # Run inner model
        y = self.inner_model(x, **kwargs)  # (B, out_t, C, H, W)

        # (B, out_t, C, H, W) → (B, H, W, out_t, C) → (B, H, W, out_t*C)
        y = y.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
        y = y.reshape(B, H, W, self.out_t * self.num_vars)

        return y
