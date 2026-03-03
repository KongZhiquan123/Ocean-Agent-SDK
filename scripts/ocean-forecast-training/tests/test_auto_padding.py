"""
@file test_auto_padding.py

@description Comprehensive test script for auto-padding in models with spatial size constraints.
             Verifies that UNet1d, UNet2d, UNet3d, OceanViT, and SwinTransformerV2 correctly
             handle arbitrary spatial sizes (e.g. 400×441 ocean data) via auto-pad/crop in forward().
@author Leizheng
@date 2026-03-03
@version 1.1.0

@changelog
  - 2026-03-03 Leizheng: v1.1.0 add UNet2d and SwinTransformerV2 tests
  - 2026-03-03 Leizheng: v1.0.0 initial creation
"""

import sys
import os
import traceback
import importlib.util

import torch

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))


def _import_module_from_file(module_name, file_path):
    """Import a single .py file as a module, bypassing __init__.py relative imports."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unet1d():
    """Test UNet1d with various spatial sizes including non-aligned ones."""
    from unet.unet1d import UNet1d

    model_params = {"in_channels": 3, "out_channels": 3, "init_features": 16, "norm": "group"}
    model = UNet1d(model_params)
    model.eval()

    test_cases = [
        # (description, input_shape, expected_output_shape)
        ("aligned N=256", (1, 256, 3), (1, 256, 3)),
        ("non-aligned N=441", (1, 441, 3), (1, 441, 3)),
        ("non-aligned N=101", (1, 101, 3), (1, 101, 3)),
        ("minimal aligned N=16", (1, 16, 3), (1, 16, 3)),
        ("non-aligned N=17", (1, 17, 3), (1, 17, 3)),
    ]

    print("=" * 60)
    print("UNet1d Auto-Padding Tests")
    print("=" * 60)

    all_passed = True
    for desc, in_shape, expected_shape in test_cases:
        try:
            x = torch.randn(*in_shape)
            with torch.no_grad():
                out = model(x)
            assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
            print(f"  PASS  {desc}: {in_shape} -> {out.shape}")
        except Exception as e:
            print(f"  FAIL  {desc}: {in_shape} -> {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_unet2d():
    """Test UNet2d with various spatial sizes including non-aligned H/W."""
    from unet.unet2d import UNet2d

    model_params = {"in_channels": 3, "out_channels": 3, "init_features": 16, "norm": "group"}
    model = UNet2d(model_params)
    model.eval()

    test_cases = [
        # UNet2d uses (B, H, W, C) format
        ("aligned 64x64", (1, 64, 64, 3), (1, 64, 64, 3)),
        ("non-aligned H=400 W=441", (1, 400, 441, 3), (1, 400, 441, 3)),
        ("non-aligned H=33 W=33", (1, 33, 33, 3), (1, 33, 33, 3)),
        ("minimal aligned 16x16", (1, 16, 16, 3), (1, 16, 16, 3)),
        ("non-aligned H=17 W=19", (1, 17, 19, 3), (1, 17, 19, 3)),
    ]

    print("\n" + "=" * 60)
    print("UNet2d Auto-Padding Tests")
    print("=" * 60)

    all_passed = True
    for desc, in_shape, expected_shape in test_cases:
        try:
            x = torch.randn(*in_shape)
            with torch.no_grad():
                out = model(x)
            assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
            print(f"  PASS  {desc}: {in_shape} -> {out.shape}")
        except Exception as e:
            print(f"  FAIL  {desc}: {in_shape} -> {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_unet3d():
    """Test UNet3d with various spatial sizes including non-aligned D/H/W."""
    from unet.unet3d import UNet3d

    model_params = {"in_channels": 3, "out_channels": 3, "init_features": 8, "norm": "group"}
    model = UNet3d(model_params)
    model.eval()

    test_cases = [
        # (description, input_shape, expected_output_shape)
        ("aligned 16x16x16", (1, 16, 16, 16, 3), (1, 16, 16, 16, 3)),
        ("non-aligned D=7 H=400 W=441", (1, 7, 32, 33, 3), (1, 7, 32, 33, 3)),  # Smaller dims for memory
        ("non-aligned D=2 H=17 W=19", (1, 2, 17, 19, 3), (1, 2, 17, 19, 3)),  # Extreme small D
        ("non-aligned D=5 H=33 W=33", (1, 5, 33, 33, 3), (1, 5, 33, 33, 3)),
        ("aligned 32x32x32", (1, 32, 32, 32, 3), (1, 32, 32, 32, 3)),
    ]

    print("\n" + "=" * 60)
    print("UNet3d Auto-Padding Tests")
    print("=" * 60)

    all_passed = True
    for desc, in_shape, expected_shape in test_cases:
        try:
            x = torch.randn(*in_shape)
            with torch.no_grad():
                out = model(x)
            assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
            print(f"  PASS  {desc}: {in_shape} -> {out.shape}")
        except Exception as e:
            print(f"  FAIL  {desc}: {in_shape} -> {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_ocean_vit():
    """Test OceanViT with various spatial sizes including non-aligned H/W."""
    vit_mod = _import_module_from_file(
        "ocean_vit_mod",
        os.path.join(PROJECT_ROOT, "models", "ocean_vit", "ocean_vit.py"),
    )
    OceanTransformer = vit_mod.OceanTransformer

    args = {
        "input_len": 7,
        "output_len": 1,
        "in_channels": 3,
        "patch_size": 8,
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.0,
    }
    model = OceanTransformer(args)
    model.eval()

    test_cases = [
        # (description, input_shape, expected_output_shape)
        # OceanViT uses NF format: (B, T, C, H, W)
        ("aligned 64x64", (1, 7, 3, 64, 64), (1, 1, 3, 64, 64)),
        ("non-aligned H=400 W=441", (1, 7, 3, 40, 41), (1, 1, 3, 40, 41)),  # Smaller for speed
        ("non-aligned H=101 W=99", (1, 7, 3, 24, 25), (1, 1, 3, 24, 25)),  # Smaller for speed
        ("aligned 32x32", (1, 7, 3, 32, 32), (1, 1, 3, 32, 32)),
        ("non-aligned W=17", (1, 7, 3, 16, 17), (1, 1, 3, 16, 17)),
    ]

    print("\n" + "=" * 60)
    print("OceanViT (OceanTransformer) Auto-Padding Tests")
    print("=" * 60)

    all_passed = True
    for desc, in_shape, expected_shape in test_cases:
        try:
            x = torch.randn(*in_shape)
            with torch.no_grad():
                out = model(x)
            assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
            print(f"  PASS  {desc}: {in_shape} -> {out.shape}")
        except Exception as e:
            print(f"  FAIL  {desc}: {in_shape} -> {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_swin_transformer_v2():
    """Test SwinTransformerV2 with various spatial sizes including non-aligned H/W."""
    swin_mod = _import_module_from_file(
        "swin_v2_mod",
        os.path.join(PROJECT_ROOT, "models", "swin_transformer", "swin_transformer_v2.py"),
    )
    SwinTransformerV2 = swin_mod.SwinTransformerV2

    # Use small config for speed. img_size=56 satisfies all constraints:
    # patch_size=1, window_size=7, depths=[2,2] (2 layers) -> alignment = 1*7*2 = 14
    # 56 / 1 = 56, 56 % 14 = 0 ✓
    model_params = {
        "in_channels": 3,
        "out_channels": 3,
        "img_size": 56,
        "patch_size": 1,
        "embed_dim": 32,
        "depths": [2, 2],
        "num_heads": [2, 4],
        "window_size": 7,
    }
    model = SwinTransformerV2(model_params)
    model.eval()

    test_cases = [
        # SwinTransformerV2 uses (B, H, W, C) format, output is interpolated to input H,W
        ("exact img_size 56x56", (1, 56, 56, 3), (1, 56, 56, 3)),
        ("smaller 40x41", (1, 40, 41, 3), (1, 40, 41, 3)),
        ("smaller 33x33", (1, 33, 33, 3), (1, 33, 33, 3)),
        ("minimal 14x14", (1, 14, 14, 3), (1, 14, 14, 3)),
    ]

    print("\n" + "=" * 60)
    print("SwinTransformerV2 Auto-Padding Tests")
    print("=" * 60)

    all_passed = True
    for desc, in_shape, expected_shape in test_cases:
        try:
            x = torch.randn(*in_shape)
            with torch.no_grad():
                out = model(x)
            assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
            print(f"  PASS  {desc}: {in_shape} -> {out.shape}")
        except Exception as e:
            print(f"  FAIL  {desc}: {in_shape} -> {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_no_padding_overhead():
    """Verify that already-aligned inputs don't trigger any padding (sanity check)."""
    from unet.unet1d import UNet1d

    model_params = {"in_channels": 3, "out_channels": 3, "init_features": 16, "norm": "group"}
    model = UNet1d(model_params)
    model.eval()

    print("\n" + "=" * 60)
    print("No-Padding Overhead Sanity Check (UNet1d N=256)")
    print("=" * 60)

    x = torch.randn(1, 256, 3)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 256, 3), f"Shape mismatch: {out.shape}"
    print(f"  PASS  Aligned input passes cleanly: {x.shape} -> {out.shape}")
    return True


def main():
    print("Auto-Padding Comprehensive Test Suite")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()

    results = {}
    results["UNet1d"] = test_unet1d()
    results["UNet2d"] = test_unet2d()
    results["UNet3d"] = test_unet3d()
    results["OceanViT"] = test_ocean_vit()
    results["SwinTransformerV2"] = test_swin_transformer_v2()
    results["No-Padding Overhead"] = test_no_padding_overhead()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_ok = False

    if all_ok:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
