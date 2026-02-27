"""
@file __init__.py

@description Package init for ocean_resnet model, wrapping raw OceanResNet with ForecastModelAdapter.
@author Leizheng
@date 2026-02-27
@version 1.0.0

@changelog
  - 2026-02-27 Leizheng: v1.0.0 initial creation
"""

from ..base.forecast_adapter import ForecastModelAdapter
from .ocean_resnet import OceanResNet as _RawOceanResNet


class OceanResNet(ForecastModelAdapter):
    def __init__(self, model_params, **kwargs):
        super().__init__(model_params, _RawOceanResNet, **kwargs)
