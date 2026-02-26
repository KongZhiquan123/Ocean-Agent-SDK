# datasets/__init__.py
# @changelog
#   - 2026-02-26 Leizheng: v1.1.0 add OceanForecastNpyDataset

from .ns2d import NS2DDataset
from .ns3d import NS3DDataset
from .carra import CarraDataset
from .airfoil_time import AirfoilTimeDataset
from .ocean_forecast_npy import OceanForecastNpyDataset


DATASET_REGISTRY = {
    'ns2d': NS2DDataset,
    'ns3d': NS3DDataset,
    'carra': CarraDataset,
    'airfoil_time': AirfoilTimeDataset,
    'ocean_forecast_npy': OceanForecastNpyDataset,
}

__all__ = [
    'DATASET_REGISTRY',
    'NS2DDataset',
    'NS3DDataset',
    'CarraDataset',
    'AirfoilTimeDataset',
    'OceanForecastNpyDataset',
]
