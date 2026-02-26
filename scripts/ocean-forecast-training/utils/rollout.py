# utils/rollout.py
from __future__ import annotations
from typing import Any, Optional, Callable
import torch


@torch.no_grad()
def autoregressive_rollout(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    u0: torch.Tensor,
    u: torch.Tensor,
    steps: int,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Generic autoregressive rollout supporting both flat and spatial tensors.

    Args:
        step_fn: one-step function u_next = step_fn(u_cur).
        u0: (B, N, C) flat or (B, *spatial, C) spatial initial field.
        u: ground truth for teacher forcing; (B, steps, ...) or None.
        steps: number of rollout steps.

    Returns:
        seq: (B, steps, N, C) or (B, steps, *spatial, C)
    """
    cur = u0
    preds = []
    for i in range(steps):
        if u is not None:
            # u shape: (B, steps, N, C) or (B, steps, *spatial, C)
            kwargs['y'] = u[:, i]
        else:
            kwargs['y'] = None
        nxt = step_fn(cur, **kwargs)
        preds.append(nxt.unsqueeze(1))
        cur = nxt
    return torch.cat(preds, dim=1)
