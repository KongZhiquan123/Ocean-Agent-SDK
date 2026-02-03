#!/usr/bin/env python3
"""
generate_static_badcases.py

Author: leizheng
Time: 2026-02-03
Description: 生成静态文件相关的 badcase 测试数据
             用于测试海洋数据预处理的防错机制

生成的文件:
- good_static.nc              正常的静态文件（基准）
- bad_static_nan.nc           静态变量(h)含 NaN
- bad_static_lon_nan.nc       经度 lon_rho 含 NaN
- bad_static_lat_nan.nc       纬度 lat_rho 含 NaN
- bad_static_lon_range.nc     经度超出 [-180, 180]
- bad_static_lat_range.nc     纬度超出 [-90, 90]
- bad_mask_not_binary.nc      掩码含 0/1/2/3 等非二值
- bad_mask_shape_mismatch.nc  掩码形状与动态数据不匹配
- bad_mask_inverted.nc        掩码反转（陆地=1，海洋=0）

Usage:
    python generate_static_badcases.py --output-dir ./work_ocean/bad_cases
"""

import argparse
import os
import numpy as np
import xarray as xr
from datetime import datetime


def create_base_static_data(lat_size: int = 100, lon_size: int = 200):
    """
    创建基础静态数据（正常数据）

    Args:
        lat_size: 纬度维度大小
        lon_size: 经度维度大小

    Returns:
        各变量的数据字典
    """
    # 创建经纬度网格
    lon_1d = np.linspace(-180, 180, lon_size, dtype=np.float32)
    lat_1d = np.linspace(-90, 90, lat_size, dtype=np.float32)
    lon_rho, lat_rho = np.meshgrid(lon_1d, lat_1d)

    # 创建海底深度 h（米），简单的梯度
    h = np.abs(lat_rho) * 50 + 100  # 100-4600m 范围
    h = h.astype(np.float32)

    # 创建掩码：简单的陆地/海洋分布
    # 假设纬度 > 85 或 < -85 是陆地（极地），其他是海洋
    mask_rho = np.ones((lat_size, lon_size), dtype=np.float32)
    mask_rho[lat_rho > 85] = 0  # 北极陆地
    mask_rho[lat_rho < -85] = 0  # 南极陆地
    # 添加一些随机陆地块
    np.random.seed(42)
    for _ in range(20):
        cx, cy = np.random.randint(10, lon_size-10), np.random.randint(10, lat_size-10)
        mask_rho[cy-5:cy+5, cx-5:cx+5] = 0

    # 创建角度 angle
    angle = np.zeros((lat_size, lon_size), dtype=np.float32)

    # u/v 网格的坐标和掩码（交错网格）
    lon_u = (lon_rho[:, :-1] + lon_rho[:, 1:]) / 2
    lat_u = (lat_rho[:, :-1] + lat_rho[:, 1:]) / 2
    mask_u = (mask_rho[:, :-1] * mask_rho[:, 1:]).astype(np.float32)

    lon_v = (lon_rho[:-1, :] + lon_rho[1:, :]) / 2
    lat_v = (lat_rho[:-1, :] + lat_rho[1:, :]) / 2
    mask_v = (mask_rho[:-1, :] * mask_rho[1:, :]).astype(np.float32)

    return {
        'lon_rho': lon_rho,
        'lat_rho': lat_rho,
        'lon_u': lon_u,
        'lat_u': lat_u,
        'lon_v': lon_v,
        'lat_v': lat_v,
        'h': h,
        'angle': angle,
        'mask_rho': mask_rho,
        'mask_u': mask_u,
        'mask_v': mask_v,
    }


def create_dynamic_data(lat_size: int = 100, lon_size: int = 200, time_steps: int = 5):
    """
    创建与静态文件匹配的动态数据

    Args:
        lat_size: 纬度维度大小
        lon_size: 经度维度大小
        time_steps: 时间步数

    Returns:
        动态数据字典
    """
    # 创建时间坐标
    time = np.arange(time_steps, dtype=np.float64) * 86400  # 每天一个时间步

    # 创建基础掩码
    base_data = create_base_static_data(lat_size, lon_size)
    mask_rho = base_data['mask_rho']

    # 创建动态变量 uo, vo
    np.random.seed(123)

    # uo: 东向流速，海洋区域有值，陆地为 0
    uo = np.random.randn(time_steps, lat_size, lon_size).astype(np.float32) * 0.5
    uo = uo * mask_rho[np.newaxis, :, :]  # 陆地区域设为 0

    # vo: 北向流速
    vo = np.random.randn(time_steps, lat_size, lon_size).astype(np.float32) * 0.5
    vo = vo * mask_rho[np.newaxis, :, :]

    return {
        'time': time,
        'uo': uo,
        'vo': vo,
        'mask_rho': mask_rho,  # 用于验证
    }


def save_static_nc(data: dict, filepath: str, description: str = ""):
    """保存静态文件为 NC 格式"""
    lat_size, lon_size = data['lon_rho'].shape

    # 创建维度
    dims_rho = ('eta_rho', 'xi_rho')
    dims_u = ('eta_rho', 'xi_u')
    dims_v = ('eta_v', 'xi_rho')

    data_vars = {}

    # rho 网格变量
    for var in ['lon_rho', 'lat_rho', 'h', 'angle', 'mask_rho']:
        if var in data:
            data_vars[var] = (dims_rho, data[var])

    # u 网格变量
    for var in ['lon_u', 'lat_u', 'mask_u']:
        if var in data:
            data_vars[var] = (dims_u, data[var])

    # v 网格变量
    for var in ['lon_v', 'lat_v', 'mask_v']:
        if var in data:
            data_vars[var] = (dims_v, data[var])

    ds = xr.Dataset(
        data_vars,
        attrs={
            'title': f'Static file for ocean preprocessing test - {description}',
            'description': description,
            'created': datetime.now().isoformat(),
            'source': 'generate_static_badcases.py'
        }
    )

    ds.to_netcdf(filepath)
    print(f"  已生成: {filepath}")
    return ds


def save_dynamic_nc(data: dict, filepath: str, description: str = ""):
    """保存动态文件为 NC 格式"""
    time_steps, lat_size, lon_size = data['uo'].shape

    ds = xr.Dataset(
        {
            'uo': (['time', 'eta_rho', 'xi_rho'], data['uo']),
            'vo': (['time', 'eta_rho', 'xi_rho'], data['vo']),
        },
        coords={
            'time': (['time'], data['time']),
        },
        attrs={
            'title': f'Dynamic file for ocean preprocessing test - {description}',
            'description': description,
            'created': datetime.now().isoformat(),
            'source': 'generate_static_badcases.py'
        }
    )

    ds.to_netcdf(filepath)
    print(f"  已生成: {filepath}")
    return ds


def generate_all_badcases(output_dir: str):
    """生成所有 badcase 文件"""
    os.makedirs(output_dir, exist_ok=True)

    lat_size, lon_size = 100, 200
    time_steps = 5

    print("=" * 60)
    print("生成静态文件相关 badcase")
    print("=" * 60)

    # 1. 生成正常的静态文件（基准）
    print("\n[1/10] 生成正常静态文件 good_static.nc")
    base_data = create_base_static_data(lat_size, lon_size)
    save_static_nc(base_data, os.path.join(output_dir, 'good_static.nc'), '正常静态文件')

    # 1.1 生成配套的动态文件
    print("\n[1.1/10] 生成配套动态文件 good_dynamic_for_static.nc")
    dyn_data = create_dynamic_data(lat_size, lon_size, time_steps)
    save_dynamic_nc(dyn_data, os.path.join(output_dir, 'good_dynamic_for_static.nc'), '配套动态文件')

    # 2. 静态变量 h 含 NaN
    print("\n[2/10] 生成 bad_static_nan.nc (静态变量h含NaN)")
    bad_data = create_base_static_data(lat_size, lon_size)
    bad_data['h'][20:30, 50:60] = np.nan  # 注入 NaN
    save_static_nc(bad_data, os.path.join(output_dir, 'bad_static_nan.nc'), '静态变量h含NaN')

    # 3. 经度 lon_rho 含 NaN
    print("\n[3/10] 生成 bad_static_lon_nan.nc (经度含NaN)")
    bad_data = create_base_static_data(lat_size, lon_size)
    bad_data['lon_rho'][10:15, 30:35] = np.nan  # 注入 NaN
    save_static_nc(bad_data, os.path.join(output_dir, 'bad_static_lon_nan.nc'), '经度lon_rho含NaN')

    # 4. 纬度 lat_rho 含 NaN
    print("\n[4/10] 生成 bad_static_lat_nan.nc (纬度含NaN)")
    bad_data = create_base_static_data(lat_size, lon_size)
    bad_data['lat_rho'][40:45, 80:85] = np.nan  # 注入 NaN
    save_static_nc(bad_data, os.path.join(output_dir, 'bad_static_lat_nan.nc'), '纬度lat_rho含NaN')

    # 5. 经度超出 [-180, 180] 范围
    print("\n[5/10] 生成 bad_static_lon_range.nc (经度超出范围)")
    bad_data = create_base_static_data(lat_size, lon_size)
    bad_data['lon_rho'][50:55, 150:155] = 200.0  # 超出范围
    save_static_nc(bad_data, os.path.join(output_dir, 'bad_static_lon_range.nc'), '经度超出[-180,180]范围')

    # 6. 纬度超出 [-90, 90] 范围
    print("\n[6/10] 生成 bad_static_lat_range.nc (纬度超出范围)")
    bad_data = create_base_static_data(lat_size, lon_size)
    bad_data['lat_rho'][60:65, 100:105] = 95.0  # 超出范围
    save_static_nc(bad_data, os.path.join(output_dir, 'bad_static_lat_range.nc'), '纬度超出[-90,90]范围')

    # 7. 掩码非二值（含 0/1/2/3）
    print("\n[7/10] 生成 bad_mask_not_binary.nc (掩码非二值)")
    bad_data = create_base_static_data(lat_size, lon_size)
    bad_data['mask_rho'][30:40, 60:70] = 2.0  # 非二值
    bad_data['mask_rho'][45:50, 80:85] = 3.0  # 非二值
    save_static_nc(bad_data, os.path.join(output_dir, 'bad_mask_not_binary.nc'), '掩码含0/1/2/3等非二值')

    # 8. 掩码形状与动态数据不匹配
    print("\n[8/10] 生成 bad_mask_shape_mismatch.nc (掩码形状不匹配)")
    # 创建不同形状的静态数据
    bad_data = create_base_static_data(lat_size + 10, lon_size + 20)  # 形状不同
    save_static_nc(bad_data, os.path.join(output_dir, 'bad_mask_shape_mismatch.nc'), '掩码形状与动态数据不匹配')

    # 9. 掩码反转（陆地=1，海洋=0）
    print("\n[9/10] 生成 bad_mask_inverted.nc (掩码反转)")
    bad_data = create_base_static_data(lat_size, lon_size)
    # 反转掩码
    bad_data['mask_rho'] = 1.0 - bad_data['mask_rho']
    bad_data['mask_u'] = 1.0 - bad_data['mask_u']
    bad_data['mask_v'] = 1.0 - bad_data['mask_v']
    save_static_nc(bad_data, os.path.join(output_dir, 'bad_mask_inverted.nc'), '掩码反转(陆地=1,海洋=0)')

    # 10. 掩码反转的配套动态数据（用于启发式验证测试）
    print("\n[10/10] 生成 bad_dynamic_for_inverted_mask.nc (配套反转掩码的动态数据)")
    # 这个动态数据使用正常掩码，但静态文件使用反转掩码
    # 这样启发式验证应该会发现掩码可能反转了
    dyn_data = create_dynamic_data(lat_size, lon_size, time_steps)
    save_dynamic_nc(dyn_data, os.path.join(output_dir, 'bad_dynamic_for_inverted_mask.nc'), '配套反转掩码测试')

    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)

    # 打印使用说明
    print(f"""
文件列表:
{output_dir}/
├── good_static.nc                    # 正常静态文件（基准）
├── good_dynamic_for_static.nc        # 配套的正常动态文件
├── bad_static_nan.nc                 # 静态变量(h)含 NaN
├── bad_static_lon_nan.nc             # 经度 lon_rho 含 NaN
├── bad_static_lat_nan.nc             # 纬度 lat_rho 含 NaN
├── bad_static_lon_range.nc           # 经度超出 [-180, 180]
├── bad_static_lat_range.nc           # 纬度超出 [-90, 90]
├── bad_mask_not_binary.nc            # 掩码含 0/1/2/3 等非二值
├── bad_mask_shape_mismatch.nc        # 掩码形状与动态数据不匹配
├── bad_mask_inverted.nc              # 掩码反转（陆地=1，海洋=0）
└── bad_dynamic_for_inverted_mask.nc  # 配套反转掩码测试

测试用例设计:
┌─────────────────────────────────────────────────────────────────────────────┐
│ 测试场景                    │ 动态文件                      │ 静态文件                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 正常流程                    │ good_dynamic_for_static.nc    │ good_static.nc             │
│ 静态变量 NaN               │ good_dynamic_for_static.nc    │ bad_static_nan.nc          │
│ 经度 NaN                   │ good_dynamic_for_static.nc    │ bad_static_lon_nan.nc      │
│ 纬度 NaN                   │ good_dynamic_for_static.nc    │ bad_static_lat_nan.nc      │
│ 经度超范围                  │ good_dynamic_for_static.nc    │ bad_static_lon_range.nc    │
│ 纬度超范围                  │ good_dynamic_for_static.nc    │ bad_static_lat_range.nc    │
│ 掩码非二值                  │ good_dynamic_for_static.nc    │ bad_mask_not_binary.nc     │
│ 掩码形状不匹配              │ good_dynamic_for_static.nc    │ bad_mask_shape_mismatch.nc │
│ 掩码反转(启发式验证)        │ bad_dynamic_for_inverted_mask │ bad_mask_inverted.nc       │
└─────────────────────────────────────────────────────────────────────────────┘
""")


def main():
    parser = argparse.ArgumentParser(description="生成静态文件相关的 badcase 测试数据")
    parser.add_argument(
        "--output-dir",
        default="./work_ocean/bad_cases",
        help="输出目录 (默认: ./work_ocean/bad_cases)"
    )
    args = parser.parse_args()

    generate_all_badcases(args.output_dir)


if __name__ == "__main__":
    main()
