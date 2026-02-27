"""
@file list_models.py

@description List available forecast models with recommendation tags.
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-27 Leizheng: v1.1.0 add 10 NeuralFramework models
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
"""

import json
import os


# All models registered in trainers/__init__.py for forecast
SUPPORTED_MODELS = [
    {
        "name": "FNO2d",
        "category": "spectral",
        "description": "Fourier Neural Operator 2D - fast, memory-efficient",
        "recommended": True,
        "notes": "Best default choice for ocean forecast; captures global patterns via FFT",
    },
    {
        "name": "UNet2d",
        "category": "cnn",
        "description": "UNet 2D - robust encoder-decoder architecture",
        "recommended": True,
        "notes": "Good for spatial detail; requires H,W divisible by 16",
    },
    {
        "name": "SwinTransformerV2",
        "category": "transformer",
        "description": "Swin Transformer V2 - shifted-window attention",
        "recommended": True,
        "notes": "Strong long-range dependency modeling; higher memory usage",
    },
    {
        "name": "Transformer",
        "category": "transformer",
        "description": "Vanilla Transformer with positional encoding",
        "recommended": False,
        "notes": "General-purpose; O(N^2) memory, best for small spatial grids",
    },
    {
        "name": "SwinMLP",
        "category": "mlp",
        "description": "Swin MLP - MLP-based shifted-window model",
        "recommended": False,
        "notes": "Lightweight alternative to SwinTransformerV2",
    },
    {
        "name": "M2NO2d",
        "category": "spectral",
        "description": "Multiplicative Multiresolution Neural Operator 2D",
        "recommended": False,
        "notes": "Multi-scale spectral operator; good for multi-resolution features",
    },
    {
        "name": "GalerkinTransformer",
        "category": "transformer",
        "description": "Galerkin Transformer - linear-complexity attention",
        "recommended": False,
        "notes": "Efficient for large grids; may need tuning",
    },
    {
        "name": "Transolver",
        "category": "transformer",
        "description": "Transolver - physics-aware transformer",
        "recommended": False,
        "notes": "Designed for PDE solving; experimental for ocean data",
    },
    {
        "name": "GNOT",
        "category": "transformer",
        "description": "General Neural Operator Transformer",
        "recommended": False,
        "notes": "Flexible operator architecture",
    },
    {
        "name": "ONO",
        "category": "operator",
        "description": "Orthogonal Neural Operator",
        "recommended": False,
        "notes": "Orthogonal decomposition-based operator",
    },
    {
        "name": "LSM",
        "category": "spectral",
        "description": "Latent Spectral Model",
        "recommended": False,
        "notes": "Latent-space spectral learning",
    },
    {
        "name": "LNO",
        "category": "spectral",
        "description": "Laplace Neural Operator",
        "recommended": False,
        "notes": "Laplace transform-based operator",
    },
    {
        "name": "MLP",
        "category": "mlp",
        "description": "Multi-Layer Perceptron baseline",
        "recommended": False,
        "notes": "Simple baseline; no spatial inductive bias",
    },
    {
        "name": "UNet1d",
        "category": "cnn",
        "description": "UNet 1D - for flattened spatial sequences",
        "recommended": False,
        "notes": "Suitable for 1D spatial domains only",
    },
    {
        "name": "FNO1d",
        "category": "spectral",
        "description": "Fourier Neural Operator 1D",
        "recommended": False,
        "notes": "Suitable for 1D spatial domains only",
    },
    {
        "name": "UNet3d",
        "category": "cnn",
        "description": "UNet 3D - encoder-decoder with depth dimension",
        "recommended": False,
        "notes": "For volumetric data; high memory usage",
    },
    {
        "name": "FNO3d",
        "category": "spectral",
        "description": "Fourier Neural Operator 3D",
        "recommended": False,
        "notes": "For volumetric data; captures 3D spectral patterns",
    },
    # NeuralFramework models
    {
        "name": "OceanCNN",
        "category": "cnn",
        "description": "Ocean CNN - basic encoder-decoder baseline",
        "recommended": False,
        "notes": "Simple baseline CNN; H,W must be divisible by 8",
    },
    {
        "name": "OceanResNet",
        "category": "cnn",
        "description": "Ocean ResNet-18 - residual encoder-decoder",
        "recommended": False,
        "notes": "ResNet-18 backbone with decoder; auto-resizes spatial output",
    },
    {
        "name": "OceanViT",
        "category": "transformer",
        "description": "Ocean Vision Transformer - patch-based ViT",
        "recommended": False,
        "notes": "Standard ViT; H,W must be divisible by patch_size (default 8)",
    },
    {
        "name": "Fuxi",
        "category": "transformer",
        "description": "Fuxi - Swin-based 2D/3D dual path weather model",
        "recommended": True,
        "notes": "Weather-adapted architecture; requires timm; strong for large grids",
    },
    {
        "name": "Fengwu",
        "category": "transformer",
        "description": "Fengwu - multi-scale 2D+3D encoder-decoder",
        "recommended": False,
        "notes": "Multi-scale architecture; high memory; requires einops",
    },
    {
        "name": "Pangu",
        "category": "transformer",
        "description": "Pangu - earth-aware attention with 2D+3D paths",
        "recommended": False,
        "notes": "Earth-specific position bias; high memory; requires einops",
    },
    {
        "name": "Crossformer",
        "category": "transformer",
        "description": "Crossformer - temporal-spatial two-stage attention",
        "recommended": True,
        "notes": "Explicit temporal modeling; good for multi-step prediction",
    },
    {
        "name": "NNG",
        "category": "graph",
        "description": "NNG - Neural Network on Graphs with icosahedron mesh",
        "recommended": False,
        "notes": "Graph-based; requires dgl + scikit-learn; high memory",
    },
    {
        "name": "OneForecast",
        "category": "graph",
        "description": "OneForecast - simplified graph neural network",
        "recommended": False,
        "notes": "Simplified graph ops; requires dgl + scikit-learn",
    },
    {
        "name": "GraphCast",
        "category": "graph",
        "description": "GraphCast - mesh-based graph neural network",
        "recommended": False,
        "notes": "Mesh-based GNN; requires dgl + scikit-learn; high memory",
    },
]


def list_models():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    available_folders = set()
    if os.path.isdir(models_dir):
        for folder in os.listdir(models_dir):
            path = os.path.join(models_dir, folder)
            if os.path.isdir(path) and not folder.startswith((".", "__")):
                available_folders.add(folder)

    result = []
    for m in SUPPORTED_MODELS:
        entry = {**m, "supported": True}
        result.append(entry)

    return result


if __name__ == "__main__":
    models = list_models()
    print(json.dumps({"models": models}, ensure_ascii=False, indent=2))
