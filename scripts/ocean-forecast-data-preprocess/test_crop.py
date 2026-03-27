#!/usr/bin/env python3
"""
测试经纬度裁剪功能
"""
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from forecast_preprocess import _compute_region_crop_indices

def test_1d_coordinates():
    """测试 1D 规则网格坐标"""
    print("测试 1D 坐标系统...")

    # 模拟经纬度：经度 100-140°E，纬度 20-50°N
    lon_arr = np.linspace(100, 140, 400)  # W 维度
    lat_arr = np.linspace(20, 50, 300)    # H 维度

    # 裁剪范围：经度 120-130°E，纬度 30-40°N
    crop_lon_range = (120.0, 130.0)
    crop_lat_range = (30.0, 40.0)

    h_start, h_end, w_start, w_end = _compute_region_crop_indices(
        lon_arr, lat_arr, crop_lon_range, crop_lat_range
    )

    print(f"  输入: lon=[{lon_arr.min():.1f}, {lon_arr.max():.1f}], lat=[{lat_arr.min():.1f}, {lat_arr.max():.1f}]")
    print(f"  裁剪: lon=[{crop_lon_range[0]}, {crop_lon_range[1]}], lat=[{crop_lat_range[0]}, {crop_lat_range[1]}]")
    print(f"  结果: H=[{h_start}:{h_end}], W=[{w_start}:{w_end}]")
    print(f"  验证: 裁剪后 lon=[{lon_arr[w_start]:.1f}, {lon_arr[w_end-1]:.1f}], lat=[{lat_arr[h_start]:.1f}, {lat_arr[h_end-1]:.1f}]")

    # 验证结果
    assert lon_arr[w_start] >= crop_lon_range[0], "经度起始点错误"
    assert lon_arr[w_end-1] <= crop_lon_range[1], "经度结束点错误"
    assert lat_arr[h_start] >= crop_lat_range[0], "纬度起始点错误"
    assert lat_arr[h_end-1] <= crop_lat_range[1], "纬度结束点错误"

    print("  ✅ 1D 坐标测试通过\n")

def test_2d_coordinates():
    """测试 2D 曲线网格坐标（如 ROMS）"""
    print("测试 2D 坐标系统...")

    # 模拟 2D 曲线网格
    h, w = 300, 400
    lon_1d = np.linspace(100, 140, w)
    lat_1d = np.linspace(20, 50, h)
    lon_arr, lat_arr = np.meshgrid(lon_1d, lat_1d)

    # 裁剪范围
    crop_lon_range = (120.0, 130.0)
    crop_lat_range = (30.0, 40.0)

    h_start, h_end, w_start, w_end = _compute_region_crop_indices(
        lon_arr, lat_arr, crop_lon_range, crop_lat_range
    )

    print(f"  输入: lon=[{lon_arr.min():.1f}, {lon_arr.max():.1f}], lat=[{lat_arr.min():.1f}, {lat_arr.max():.1f}]")
    print(f"  裁剪: lon=[{crop_lon_range[0]}, {crop_lon_range[1]}], lat=[{crop_lat_range[0]}, {crop_lat_range[1]}]")
    print(f"  结果: H=[{h_start}:{h_end}], W=[{w_start}:{w_end}]")

    # 验证结果
    cropped_lon = lon_arr[h_start:h_end, w_start:w_end]
    cropped_lat = lat_arr[h_start:h_end, w_start:w_end]

    print(f"  验证: 裁剪后 lon=[{cropped_lon.min():.1f}, {cropped_lon.max():.1f}], lat=[{cropped_lat.min():.1f}, {cropped_lat.max():.1f}]")

    assert cropped_lon.min() >= crop_lon_range[0], "经度最小值错误"
    assert cropped_lon.max() <= crop_lon_range[1], "经度最大值错误"
    assert cropped_lat.min() >= crop_lat_range[0], "纬度最小值错误"
    assert cropped_lat.max() <= crop_lat_range[1], "纬度最大值错误"

    print("  ✅ 2D 坐标测试通过\n")

if __name__ == "__main__":
    try:
        test_1d_coordinates()
        test_2d_coordinates()
        print("=" * 50)
        print("✅ 所有测试通过！经纬度裁剪功能正常工作。")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
