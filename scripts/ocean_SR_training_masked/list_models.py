"""
list_models.py - 列出所有可用的超分辨率模型

输出 JSON 格式的模型信息，供 TypeScript 工具调用。
不导入实际模型类，避免 GPU/依赖问题。
"""

import json

# 模型注册表（与 models/__init__.py 和 trainers/__init__.py 保持同步）
MODELS = [
    {"name": "FNO2d",                "category": "standard",  "trainer": "BaseTrainer",     "description": "Fourier Neural Operator 2D"},
    {"name": "UNet2d",               "category": "standard",  "trainer": "BaseTrainer",     "description": "UNet 2D"},
    {"name": "M2NO2d",               "category": "standard",  "trainer": "BaseTrainer",     "description": "Multiplicative Multiresolution Neural Operator 2D"},
    {"name": "Galerkin_Transformer", "category": "standard",  "trainer": "BaseTrainer",     "description": "Galerkin Transformer"},
    {"name": "MWT2d",                "category": "standard",  "trainer": "BaseTrainer",     "description": "Morlet Wavelet Transform 2D"},
    {"name": "SRNO",                 "category": "standard",  "trainer": "BaseTrainer",     "description": "Super-Resolution Neural Operator"},
    {"name": "Swin_Transformer",     "category": "standard",  "trainer": "BaseTrainer",     "description": "Swin Transformer SR"},
    {"name": "EDSR",                 "category": "standard",  "trainer": "BaseTrainer",     "description": "Enhanced Deep Super-Resolution"},
    {"name": "HiNOTE",               "category": "standard",  "trainer": "BaseTrainer",     "description": "High-order Neural Operator"},
    {"name": "SwinIR",               "category": "standard",  "trainer": "BaseTrainer",     "description": "SwinIR Super-Resolution"},
    {"name": "DDPM",                 "category": "diffusion", "trainer": "DDPMTrainer",     "description": "Denoising Diffusion Probabilistic Model"},
    {"name": "SR3",                  "category": "diffusion", "trainer": "DDPMTrainer",     "description": "SR3 Diffusion Model"},
    {"name": "MG-DDPM",              "category": "diffusion", "trainer": "DDPMTrainer",     "description": "Multigrid DDPM"},
    {"name": "Resshift",             "category": "diffusion", "trainer": "ResshiftTrainer", "description": "Residual Shifting Diffusion"},
    {"name": "ReMiG",                "category": "diffusion", "trainer": "ReMiGTrainer",    "description": "ReMiG Diffusion Model"},
]


def list_models():
    return MODELS


if __name__ == "__main__":
    result = list_models()
    print(json.dumps({"models": result}, ensure_ascii=False, indent=2))
