"""
@file __init__.py

@description Fuxi model package - 2D/3D dual path weather prediction model.
@author Leizheng
@date 2026-02-27
@version 1.0.0

@changelog
  - 2026-02-27 Leizheng: v1.0.0 initial creation
"""

from ..base.forecast_adapter import ForecastModelAdapter
from .fuxi import Fuxi as _RawFuxi


class Fuxi(ForecastModelAdapter):
    def __init__(self, model_params, **kwargs):
        super().__init__(model_params, _RawFuxi, **kwargs)
