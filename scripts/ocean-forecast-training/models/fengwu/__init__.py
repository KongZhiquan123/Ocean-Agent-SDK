"""
@file __init__.py
@description Fengwu model package.
@author Leizheng
@date 2026-02-27
@version 1.0.0
@changelog
  - 2026-02-27 Leizheng: v1.0.0 initial creation
"""
from ..base.forecast_adapter import ForecastModelAdapter
from .fengwu import OceanFengwu as _RawFengwu

class Fengwu(ForecastModelAdapter):
    def __init__(self, model_params, **kwargs):
        super().__init__(model_params, _RawFengwu, **kwargs)
