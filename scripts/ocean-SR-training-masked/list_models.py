"""
@file list_models.py

@description 列出所有可用/未接入的超分辨率模型（输出 JSON 供工具调用）
@author Leizheng
@date 2026-02-09
@version 1.2.0

@changelog
  - 2026-02-09 Leizheng: v1.2.0 扫描 models 目录并标记 supported/notes
  - 2026-02-09 Leizheng: v1.1.0 扩充扩散与残差扩散模型描述
  - 原始版本: v1.0.0
"""

import json
import os

# 模型注册表（与 models/__init__.py 和 trainers/__init__.py 保持同步）
SUPPORTED_MODELS = [
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

FOLDER_TO_MODEL = {
    "fno": "FNO2d",
    "unet": "UNet2d",
    "m2no": "M2NO2d",
    "galerkin": "Galerkin_Transformer",
    "MWT": "MWT2d",
    "sronet": "SRNO",
    "swin_Transformer": "Swin_Transformer",
    "EDSR": "EDSR",
    "HiNOTE": "HiNOTE",
    "swinIR": "SwinIR",
    "ddpm": "DDPM",
    "sr3": "SR3",
    "mg_ddpm": "MG-DDPM",
    "remig": "ReMiG",
    "resshift": "Resshift",
}

EXCLUDED_DIRS = {"__pycache__", "resshift copy"}


def list_models():
    models = [{**m, "supported": True} for m in SUPPORTED_MODELS]

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if os.path.isdir(models_dir):
        for folder in sorted(os.listdir(models_dir)):
            path = os.path.join(models_dir, folder)
            if not os.path.isdir(path):
                continue
            if folder.startswith(".") or folder.startswith("__"):
                continue
            if folder in EXCLUDED_DIRS:
                continue

            mapped_name = FOLDER_TO_MODEL.get(folder)
            if mapped_name and any(m["name"] == mapped_name for m in SUPPORTED_MODELS):
                continue

            models.append({
                "name": mapped_name or folder,
                "category": "experimental",
                "trainer": "N/A",
                "description": "目录存在但未接入训练流程",
                "supported": False,
                "notes": "缺少模型注册/Trainer/配置模板",
            })

    return models


if __name__ == "__main__":
    result = list_models()
    print(json.dumps({"models": result}, ensure_ascii=False, indent=2))
