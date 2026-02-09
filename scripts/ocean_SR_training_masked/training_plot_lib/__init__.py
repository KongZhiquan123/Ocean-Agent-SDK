"""
@file __init__.py

@description Unified exports for training_plot_lib package.
@author kongzhiquan
@date 2026-02-09
@version 1.0.0

@changelog
    - 2026-02-09 kongzhiquan: v1.0.0 initial package creation
"""

# Import style module to ensure matplotlib configuration is applied on import
from . import style  # noqa: F401

from .log_parser import parse_structured_log
from .plot_loss import plot_loss_curve
from .plot_metrics import plot_metrics_curve
from .plot_lr import plot_lr_curve
from .plot_comparison import plot_metrics_comparison
from .plot_sample import plot_sample_comparison
from .plot_summary import plot_training_summary

__all__ = [
    'parse_structured_log',
    'plot_loss_curve',
    'plot_metrics_curve',
    'plot_lr_curve',
    'plot_metrics_comparison',
    'plot_sample_comparison',
    'plot_training_summary',
]
