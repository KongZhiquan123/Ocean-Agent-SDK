"""
@file __init__.py
@description Crossformer model package.
@author Leizheng
@date 2026-02-27
@version 1.0.0
@changelog
  - 2026-02-27 Leizheng: v1.0.0 initial creation
"""
from ..base.forecast_adapter import ForecastModelAdapter
from .crossformer import OceanCrossformer as _RawCrossformer

class Crossformer(ForecastModelAdapter):
    def __init__(self, model_params, **kwargs):
        super().__init__(model_params, _RawCrossformer, **kwargs)
