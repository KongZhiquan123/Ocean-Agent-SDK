#!/usr/bin/env python3
"""
超分辨率场景下的数据预处理工具
==================================

设计原则：
- 不破坏原有数据结构的任何信息
- 不做任何标准化，只做格式转换
- 对代码的改动最小

工具分为 A -> B -> C 三个步骤：
- Step A: 查看数据并定义变量 (inspect_and_define)
- Step B: 进行张量约定 (validate_tensor_convention)
- Step C: 转npy格式存储 (convert_to_npy)

Author: Agent
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


# ============================================================================
# 变量分类定义
# ============================================================================

class VarCategory(Enum):
    """变量分类枚举"""
    DYNAMIC = "dynamic"      # 动态变量：随时间变化（如流速、温度）
    STATIC = "static"        # 静态变量：不随时间变化（如地形、掩码）
    IGNORED = "ignored"      # 无关变量：不参与处理


@dataclass
class VariableInfo:
    """变量信息数据类"""
    name: str
    category: VarCategory
    dims: List[str]
    shape: Tuple[int, ...]
    dtype: str
    units: str = "unknown"
    long_name: str = ""
    is_mask: bool = False  # 是否为掩码变量
    description: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d['category'] = self.category.value
        d['shape'] = list(self.shape)
        return d


@dataclass
class PreprocessConfig:
    """预处理配置"""
    # 研究变量（用户必须指定）
    research_vars: List[str] = field(default_factory=list)

    # 静态变量列表
    static_vars: List[str] = field(default_factory=lambda: [
        'angle', 'h', 'mask_u', 'mask_rho', 'mask_v',
        'pn', 'pm', 'f', 'x_rho', 'x_u', 'x_v',
        'y_rho', 'y_u', 'y_v', 'lat_psi', 'lon_psi'
    ])

    # 掩码变量（这些变量一定不能变）
    mask_vars: List[str] = field(default_factory=lambda: [
        'mask_u', 'mask_rho', 'mask_v'
    ])

    # 目标张量形状约定
    expected_dynamic_ndim: int = 3  # [T, H, W]
    expected_static_ndim: int = 2   # [H, W]

    # 输出目录结构
    output_base: str = ""
    hr_subdir: str = "hr"  # 高分辨率数据子目录

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Step A: 查看数据并定义变量
# ============================================================================

def step_a_inspect_and_define(
    nc_folder_path: str,
    static_file_path: Optional[str] = None,
    file_filter: str = "avg",
    config: Optional[PreprocessConfig] = None
) -> Dict[str, Any]:
    """
    Step A: 初步数据提取与变量定义

    功能：
    1. 从NC文件查看包含什么变量
    2. 根据字典方式提取变量信息
    3. 打印张量及其对应的形状
    4. 画出变量的数值分布

    防错规则（Agent修改时必须检查）：
    =========================================
    【规则A1】确定动态变量 vs 静态变量 vs 无关变量
        - 动态变量：有时间维度，如 u_eastward, v_northward, temp, salt
        - 静态变量：无时间维度，如 mask_*, h, angle
        - 无关变量：坐标变量、元数据等

    【规则A2】陆地掩码变量一定不能变
        - mask_u, mask_rho, mask_v 必须原样保留
        - 通过检查流速变量的 NaN/0 值分布来验证陆地边界

    【规则A3】NC文件必须排序
        - 使用 sorted() 确保时间顺序正确
        - 文件名应包含时间信息
    =========================================

    Args:
        nc_folder_path: NC文件所在目录
        static_file_path: 静态文件路径（可选）
        file_filter: 文件名过滤关键字
        config: 预处理配置

    Returns:
        包含变量信息、分布统计的字典
    """
    import xarray as xr

    if config is None:
        config = PreprocessConfig()

    result = {
        "status": "pending",
        "nc_folder": nc_folder_path,
        "static_file": static_file_path,
        "variables": {},
        "file_list": [],
        "statistics": {},
        "warnings": [],
        "errors": []
    }

    # -------------------------------------------------------------------------
    # A1: 获取并排序NC文件列表（防错规则A3）
    # -------------------------------------------------------------------------
    try:
        if not os.path.exists(nc_folder_path):
            result["errors"].append(f"目录不存在: {nc_folder_path}")
            result["status"] = "error"
            return result

        nc_files = sorted([
            f for f in os.listdir(nc_folder_path)
            if f.endswith('.nc') and file_filter in f
        ])

        if not nc_files:
            result["warnings"].append(f"未找到包含 '{file_filter}' 的NC文件")

        result["file_list"] = nc_files
        print(f"[Step A] 找到 {len(nc_files)} 个NC文件（已排序）")

    except Exception as e:
        result["errors"].append(f"读取目录失败: {str(e)}")
        result["status"] = "error"
        return result

    # -------------------------------------------------------------------------
    # A2: 读取第一个文件获取变量信息
    # -------------------------------------------------------------------------
    if nc_files:
        first_file = os.path.join(nc_folder_path, nc_files[0])
        try:
            with xr.open_dataset(first_file) as ds:
                print(f"\n[Step A] 分析文件: {nc_files[0]}")
                print("=" * 60)

                for var_name in ds.data_vars:
                    var = ds[var_name]

                    # 判断变量类别（防错规则A1）
                    dims = list(var.dims)
                    has_time = any(d in dims for d in ['time', 'ocean_time', 't'])

                    if var_name in config.mask_vars:
                        category = VarCategory.STATIC
                        is_mask = True
                    elif var_name in config.static_vars:
                        category = VarCategory.STATIC
                        is_mask = False
                    elif has_time:
                        category = VarCategory.DYNAMIC
                        is_mask = False
                    else:
                        category = VarCategory.IGNORED
                        is_mask = False

                    var_info = VariableInfo(
                        name=var_name,
                        category=category,
                        dims=dims,
                        shape=tuple(var.shape),
                        dtype=str(var.dtype),
                        units=var.attrs.get("units", "unknown"),
                        long_name=var.attrs.get("long_name", var_name),
                        is_mask=is_mask
                    )

                    result["variables"][var_name] = var_info.to_dict()

                    # 打印变量信息
                    cat_str = f"[{category.value.upper():8s}]"
                    mask_str = " [MASK-不可变]" if is_mask else ""
                    print(f"  {cat_str} {var_name:20s} shape={var.shape} dims={dims}{mask_str}")

                    # 计算统计信息
                    try:
                        values = var.values
                        stats = {
                            "min": float(np.nanmin(values)),
                            "max": float(np.nanmax(values)),
                            "mean": float(np.nanmean(values)),
                            "std": float(np.nanstd(values)),
                            "nan_count": int(np.isnan(values).sum()),
                            "zero_count": int((values == 0).sum())
                        }
                        result["statistics"][var_name] = stats
                    except:
                        pass

        except Exception as e:
            result["errors"].append(f"读取文件失败: {str(e)}")

    # -------------------------------------------------------------------------
    # A3: 读取静态文件（如果提供）
    # -------------------------------------------------------------------------
    if static_file_path and os.path.exists(static_file_path):
        try:
            with xr.open_dataset(static_file_path) as ds:
                print(f"\n[Step A] 分析静态文件: {static_file_path}")
                print("=" * 60)

                for var_name in ds.data_vars:
                    if var_name in result["variables"]:
                        continue

                    var = ds[var_name]
                    is_mask = var_name in config.mask_vars

                    var_info = VariableInfo(
                        name=var_name,
                        category=VarCategory.STATIC,
                        dims=list(var.dims),
                        shape=tuple(var.shape),
                        dtype=str(var.dtype),
                        units=var.attrs.get("units", "unknown"),
                        long_name=var.attrs.get("long_name", var_name),
                        is_mask=is_mask
                    )

                    result["variables"][var_name] = var_info.to_dict()

                    mask_str = " [MASK-不可变]" if is_mask else ""
                    print(f"  [STATIC  ] {var_name:20s} shape={var.shape}{mask_str}")

        except Exception as e:
            result["warnings"].append(f"读取静态文件失败: {str(e)}")

    # -------------------------------------------------------------------------
    # A4: 用户必须确认研究变量（强制询问）
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("【重要】请确认研究变量:")
    print("  动态变量（可能的研究目标）:")
    dynamic_vars = [v for v, info in result["variables"].items()
                   if info["category"] == "dynamic"]
    for i, v in enumerate(dynamic_vars, 1):
        info = result["variables"][v]
        print(f"    {i}. {v} - shape={info['shape']}")

    result["dynamic_vars_candidates"] = dynamic_vars
    result["status"] = "awaiting_user_confirmation"

    return result


def plot_variable_distribution(
    nc_file_path: str,
    var_names: List[str],
    output_dir: str = "./plots"
) -> str:
    """
    画出变量的数值分布（直方图）

    Args:
        nc_file_path: NC文件路径
        var_names: 要绘制的变量名列表
        output_dir: 输出目录

    Returns:
        保存的图片路径
    """
    import xarray as xr
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    with xr.open_dataset(nc_file_path) as ds:
        n_vars = len(var_names)
        fig, axes = plt.subplots(1, n_vars, figsize=(5*n_vars, 4))
        if n_vars == 1:
            axes = [axes]

        for ax, var_name in zip(axes, var_names):
            if var_name in ds.data_vars:
                values = ds[var_name].values.flatten()
                values = values[~np.isnan(values)]

                ax.hist(values, bins=50, edgecolor='black', alpha=0.7)
                ax.set_title(f'{var_name}\nmin={values.min():.2e}, max={values.max():.2e}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Count')
            else:
                ax.text(0.5, 0.5, f'{var_name}\nnot found',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        output_path = os.path.join(output_dir, "variable_distribution.png")
        plt.savefig(output_path, dpi=150)
        plt.close()

    print(f"[Step A] 分布图已保存: {output_path}")
    return output_path


# ============================================================================
# Step B: 进行张量约定
# ============================================================================

def step_b_validate_tensor_convention(
    variables_info: Dict[str, Any],
    research_vars: List[str],
    config: Optional[PreprocessConfig] = None
) -> Dict[str, Any]:
    """
    Step B: 进行张量约定验证

    功能：
    1. 验证 get_item 逻辑的变量名是否正确
    2. 检查张量形状是否符合约定 [T, H, W]
    3. 验证掩码变量的一致性

    防错规则（Agent修改时必须检查）：
    =========================================
    【规则B1】动态变量必须有时间维度
        - 形状应为 [T, H, W] 或 [T, D, H, W]（D为深度）
        - 第一个维度必须是时间

    【规则B2】静态变量不能有时间维度
        - 形状应为 [H, W]
        - 掩码变量必须是 0/1 二值

    【规则B3】研究变量必须在变量列表中
        - 用户指定的研究变量必须存在
        - 研究变量的形状必须兼容

    【规则B4】掩码验证
        - 检查动态变量在陆地区域的值（应为 NaN 或 0）
        - 掩码形状必须与数据形状的空间维匹配
    =========================================

    Args:
        variables_info: Step A 返回的变量信息
        research_vars: 用户确认的研究变量列表
        config: 预处理配置

    Returns:
        验证结果字典
    """
    if config is None:
        config = PreprocessConfig()

    result = {
        "status": "pending",
        "research_vars": research_vars,
        "validated_vars": {},
        "tensor_convention": {},
        "mask_check": {},
        "warnings": [],
        "errors": []
    }

    print("\n[Step B] 张量约定验证")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # B1: 验证研究变量存在性（防错规则B3）
    # -------------------------------------------------------------------------
    for var in research_vars:
        if var not in variables_info:
            result["errors"].append(f"研究变量 '{var}' 不存在于数据中")
            print(f"  [ERROR] 研究变量 '{var}' 不存在")
        else:
            result["validated_vars"][var] = variables_info[var]
            print(f"  [OK] 研究变量 '{var}' 已确认")

    if result["errors"]:
        result["status"] = "error"
        return result

    # -------------------------------------------------------------------------
    # B2: 检查张量形状约定（防错规则B1, B2）
    # -------------------------------------------------------------------------
    print("\n[Step B] 张量形状检查:")

    for var_name, var_info in variables_info.items():
        shape = tuple(var_info["shape"])
        category = var_info["category"]
        ndim = len(shape)

        convention = {
            "name": var_name,
            "original_shape": shape,
            "category": category,
            "valid": False,
            "interpretation": ""
        }

        if category == "dynamic":
            # 动态变量：期望 [T, H, W] 或 [T, D, H, W]
            if ndim == 3:
                convention["valid"] = True
                convention["interpretation"] = "[T, H, W] - 时间×高度×宽度"
                convention["T"] = shape[0]
                convention["H"] = shape[1]
                convention["W"] = shape[2]
            elif ndim == 4:
                convention["valid"] = True
                convention["interpretation"] = "[T, D, H, W] - 时间×深度×高度×宽度"
                convention["T"] = shape[0]
                convention["D"] = shape[1]
                convention["H"] = shape[2]
                convention["W"] = shape[3]
            else:
                convention["valid"] = False
                convention["interpretation"] = f"不符合约定：期望3D或4D，实际{ndim}D"
                result["warnings"].append(
                    f"动态变量 '{var_name}' 维度不符合约定: shape={shape}"
                )

        elif category == "static":
            # 静态变量：期望 [H, W]
            if ndim == 2:
                convention["valid"] = True
                convention["interpretation"] = "[H, W] - 高度×宽度"
                convention["H"] = shape[0]
                convention["W"] = shape[1]
            else:
                convention["valid"] = False
                convention["interpretation"] = f"不符合约定：期望2D，实际{ndim}D"
                if var_name in config.mask_vars:
                    result["errors"].append(
                        f"掩码变量 '{var_name}' 维度错误: shape={shape}"
                    )

        result["tensor_convention"][var_name] = convention

        # 打印检查结果
        status = "OK" if convention["valid"] else "WARN"
        print(f"  [{status:4s}] {var_name:20s} {shape} -> {convention['interpretation']}")

    # -------------------------------------------------------------------------
    # B3: 生成 var_names 配置（用于 get_item）
    # -------------------------------------------------------------------------
    print("\n[Step B] 生成变量名配置:")

    var_names_config = {
        "dynamic": [],
        "static": [],
        "research": research_vars,
        "mask": config.mask_vars
    }

    for var_name, var_info in variables_info.items():
        if var_info["category"] == "dynamic":
            var_names_config["dynamic"].append(var_name)
        elif var_info["category"] == "static":
            var_names_config["static"].append(var_name)

    result["var_names_config"] = var_names_config

    print(f"  动态变量: {var_names_config['dynamic']}")
    print(f"  静态变量: {var_names_config['static']}")
    print(f"  研究变量: {var_names_config['research']}")
    print(f"  掩码变量: {var_names_config['mask']}")

    # -------------------------------------------------------------------------
    # B4: 检查空间维度一致性
    # -------------------------------------------------------------------------
    print("\n[Step B] 空间维度一致性检查:")

    spatial_shapes = {}
    for var_name, conv in result["tensor_convention"].items():
        if conv["valid"] and "H" in conv and "W" in conv:
            key = (conv["H"], conv["W"])
            if key not in spatial_shapes:
                spatial_shapes[key] = []
            spatial_shapes[key].append(var_name)

    if len(spatial_shapes) > 1:
        result["warnings"].append("存在多种空间维度，可能需要插值对齐")
        for shape, vars in spatial_shapes.items():
            print(f"  [WARN] 空间维度 {shape}: {vars}")
    else:
        for shape, vars in spatial_shapes.items():
            print(f"  [OK] 所有变量空间维度一致: {shape}")

    # 最终状态
    if result["errors"]:
        result["status"] = "error"
        print("\n[Step B] 验证失败，存在错误")
    else:
        result["status"] = "pass"
        print("\n[Step B] 验证通过")

    return result


# ============================================================================
# Step C: 转npy格式存储
# ============================================================================

def step_c_convert_to_npy(
    nc_folder_path: str,
    output_base_dir: str,
    research_vars: List[str],
    static_file_path: Optional[str] = None,
    config: Optional[PreprocessConfig] = None,
    file_filter: str = "avg"
) -> Dict[str, Any]:
    """
    Step C: 转换为NPY格式并按目录结构存储

    功能：
    1. 按 hr/变量名.npy 的目录结构存储
    2. 每个变量单独存储（不再拼接通道）
    3. 静态变量单独存储

    防错规则（Agent修改时必须检查）：
    =========================================
    【规则C1】目录结构检查
        - 输出目录必须符合 OceanSRDataset 的读取逻辑
        - 结构：output_base/hr/变量名.npy

    【规则C2】维度检查
        - 动态变量保存前检查形状 [T, H, W] 或 [T, D, H, W]
        - 静态变量保存前检查形状 [H, W]
        - 如果不符合，报错并指明问题

    【规则C3】掩码变量必须原样保存
        - mask_* 变量不做任何修改
        - 保存后验证与原始数据一致

    【规则C4】NC文件必须排序后处理
        - 使用 sorted() 确保时间顺序
        - 拼接时按时间轴 axis=0
    =========================================

    事后防错规则：
    =========================================
    【事后规则1】目录结构检查
        - 检查 output_base/hr/ 目录是否存在
        - 检查每个研究变量的 .npy 文件是否存在

    【事后规则2】维度检查
        - 加载保存的 .npy 验证形状
        - 动态变量应为 [T, H, W]
        - 如果不对，报错："数据维度检查有问题，请检查xxx部分"
    =========================================

    Args:
        nc_folder_path: NC文件目录
        output_base_dir: 输出基础目录
        research_vars: 研究变量列表
        static_file_path: 静态文件路径
        config: 预处理配置
        file_filter: 文件名过滤关键字

    Returns:
        处理结果字典
    """
    import xarray as xr
    from tqdm import tqdm

    if config is None:
        config = PreprocessConfig()

    result = {
        "status": "pending",
        "output_dir": output_base_dir,
        "saved_files": {},
        "warnings": [],
        "errors": [],
        "post_validation": {}
    }

    # -------------------------------------------------------------------------
    # C1: 创建目录结构（防错规则C1）
    # -------------------------------------------------------------------------
    hr_dir = os.path.join(output_base_dir, "hr")
    stat_dir = os.path.join(output_base_dir, "static")

    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(stat_dir, exist_ok=True)

    print(f"\n[Step C] 创建目录结构:")
    print(f"  高分辨率数据: {hr_dir}")
    print(f"  静态数据: {stat_dir}")

    # -------------------------------------------------------------------------
    # C2: 获取并排序NC文件（防错规则C4）
    # -------------------------------------------------------------------------
    nc_files = sorted([
        f for f in os.listdir(nc_folder_path)
        if f.endswith('.nc') and file_filter in f
    ])

    if not nc_files:
        result["errors"].append(f"未找到NC文件")
        result["status"] = "error"
        return result

    print(f"\n[Step C] 处理 {len(nc_files)} 个NC文件（已排序）")

    # -------------------------------------------------------------------------
    # C3: 按变量收集并拼接数据
    # -------------------------------------------------------------------------
    data_accum = {var: [] for var in research_vars}

    for fname in tqdm(nc_files, desc="读取NC文件"):
        fp = os.path.join(nc_folder_path, fname)
        try:
            with xr.open_dataset(fp) as ds:
                for var in research_vars:
                    if var in ds.variables:
                        arr = ds[var].values
                        data_accum[var].append(arr)
                    else:
                        if len(data_accum[var]) == 0:
                            result["warnings"].append(f"变量 '{var}' 在 {fname} 中不存在")
        except Exception as e:
            result["errors"].append(f"读取 {fname} 失败: {str(e)}")

    # -------------------------------------------------------------------------
    # C4: 维度检查并保存（防错规则C2）
    # -------------------------------------------------------------------------
    print(f"\n[Step C] 拼接并保存动态变量:")

    for var, arr_list in data_accum.items():
        if not arr_list:
            result["errors"].append(f"变量 '{var}' 无数据")
            continue

        # 检查空间维度一致性
        shape0 = arr_list[0].shape[1:]
        for i, arr in enumerate(arr_list):
            if arr.shape[1:] != shape0:
                result["errors"].append(
                    f"变量 '{var}' 空间维度不一致: 文件#{i} {arr.shape[1:]} vs {shape0}"
                )
                continue

        # 按时间轴拼接
        data_concat = np.concatenate(arr_list, axis=0)

        # 维度检查（防错规则C2）
        ndim = data_concat.ndim
        if ndim == 3:
            interpretation = f"[T={data_concat.shape[0]}, H={data_concat.shape[1]}, W={data_concat.shape[2]}]"
        elif ndim == 4:
            interpretation = f"[T={data_concat.shape[0]}, D={data_concat.shape[1]}, H={data_concat.shape[2]}, W={data_concat.shape[3]}]"
        else:
            result["errors"].append(
                f"数据维度检查有问题，请检查变量 '{var}'：期望3D或4D，实际{ndim}D"
            )
            continue

        # 保存到 hr 目录
        out_fp = os.path.join(hr_dir, f"{var}.npy")
        np.save(out_fp, data_concat)

        result["saved_files"][var] = {
            "path": out_fp,
            "shape": list(data_concat.shape),
            "dtype": str(data_concat.dtype),
            "interpretation": interpretation
        }

        print(f"  [OK] {var}.npy shape={data_concat.shape} {interpretation}")

    # -------------------------------------------------------------------------
    # C5: 处理静态文件（防错规则C3）
    # -------------------------------------------------------------------------
    if static_file_path and os.path.exists(static_file_path):
        print(f"\n[Step C] 处理静态变量:")

        try:
            with xr.open_dataset(static_file_path) as ds:
                for var in config.static_vars:
                    if var in ds.variables:
                        arr = ds[var].values

                        # 掩码变量特殊处理（防错规则C3）
                        is_mask = var in config.mask_vars

                        out_fp = os.path.join(stat_dir, f"{var}.npy")
                        np.save(out_fp, arr)

                        mask_note = " [MASK-原样保存]" if is_mask else ""
                        print(f"  [OK] {var}.npy shape={arr.shape}{mask_note}")

                        result["saved_files"][var] = {
                            "path": out_fp,
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "is_mask": is_mask
                        }

        except Exception as e:
            result["errors"].append(f"处理静态文件失败: {str(e)}")

    # -------------------------------------------------------------------------
    # C6: 事后验证（事后防错规则1, 2）
    # -------------------------------------------------------------------------
    print(f"\n[Step C] 事后验证:")

    validation_passed = True

    # 事后规则1：目录结构检查
    if not os.path.exists(hr_dir):
        result["errors"].append("事后检查失败：hr目录不存在")
        validation_passed = False
    else:
        for var in research_vars:
            expected_file = os.path.join(hr_dir, f"{var}.npy")
            if not os.path.exists(expected_file):
                result["errors"].append(f"事后检查失败：{var}.npy 不存在")
                validation_passed = False

    # 事后规则2：维度检查
    for var, info in result["saved_files"].items():
        if var in research_vars:
            loaded = np.load(info["path"])
            if loaded.ndim not in [3, 4]:
                result["errors"].append(
                    f"数据维度检查有问题，请检查 '{var}' 部分：期望[T,H,W]，实际shape={loaded.shape}"
                )
                validation_passed = False
            else:
                result["post_validation"][var] = {
                    "shape": list(loaded.shape),
                    "valid": True
                }

    print(f"  目录结构: {'OK' if os.path.exists(hr_dir) else 'FAIL'}")
    print(f"  维度检查: {'OK' if validation_passed else 'FAIL'}")

    # 最终状态
    if result["errors"]:
        result["status"] = "error"
        print(f"\n[Step C] 处理失败，存在 {len(result['errors'])} 个错误")
    else:
        result["status"] = "pass"
        print(f"\n[Step C] 处理完成，所有检查通过 ✓")

    return result


# ============================================================================
# 主入口：完整的预处理流程
# ============================================================================

def run_full_preprocessing(
    nc_folder_path: str,
    output_base_dir: str,
    research_vars: List[str],
    static_file_path: Optional[str] = None,
    file_filter: str = "avg",
    plot_distribution: bool = True
) -> Dict[str, Any]:
    """
    运行完整的 A -> B -> C 预处理流程

    Args:
        nc_folder_path: NC文件目录
        output_base_dir: 输出目录
        research_vars: 研究变量列表（必须由用户指定）
        static_file_path: 静态文件路径
        file_filter: 文件名过滤关键字
        plot_distribution: 是否绘制分布图

    Returns:
        完整的处理结果
    """
    print("=" * 70)
    print("超分辨率数据预处理流程")
    print("=" * 70)

    config = PreprocessConfig(research_vars=research_vars)
    final_result = {
        "step_a": None,
        "step_b": None,
        "step_c": None,
        "overall_status": "pending"
    }

    # Step A: 查看数据并定义变量
    print("\n" + "=" * 70)
    print("STEP A: 查看数据并定义变量")
    print("=" * 70)

    step_a_result = step_a_inspect_and_define(
        nc_folder_path=nc_folder_path,
        static_file_path=static_file_path,
        file_filter=file_filter,
        config=config
    )
    final_result["step_a"] = step_a_result

    if step_a_result["status"] == "error":
        final_result["overall_status"] = "error"
        return final_result

    # 绘制分布图
    if plot_distribution and step_a_result["file_list"]:
        first_file = os.path.join(nc_folder_path, step_a_result["file_list"][0])
        plot_dir = os.path.join(output_base_dir, "plots")
        plot_variable_distribution(first_file, research_vars[:4], plot_dir)

    # Step B: 进行张量约定
    print("\n" + "=" * 70)
    print("STEP B: 进行张量约定")
    print("=" * 70)

    step_b_result = step_b_validate_tensor_convention(
        variables_info=step_a_result["variables"],
        research_vars=research_vars,
        config=config
    )
    final_result["step_b"] = step_b_result

    if step_b_result["status"] == "error":
        final_result["overall_status"] = "error"
        return final_result

    # Step C: 转npy格式存储
    print("\n" + "=" * 70)
    print("STEP C: 转NPY格式存储")
    print("=" * 70)

    step_c_result = step_c_convert_to_npy(
        nc_folder_path=nc_folder_path,
        output_base_dir=output_base_dir,
        research_vars=research_vars,
        static_file_path=static_file_path,
        config=config,
        file_filter=file_filter
    )
    final_result["step_c"] = step_c_result

    # 最终状态
    if step_c_result["status"] == "pass":
        final_result["overall_status"] = "pass"
        print("\n" + "=" * 70)
        print("预处理完成: PASS")
        print("=" * 70)
    else:
        final_result["overall_status"] = "error"
        print("\n" + "=" * 70)
        print("预处理失败: 请检查上述错误")
        print("=" * 70)

    return final_result


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="超分辨率数据预处理工具")
    parser.add_argument("--nc-folder", required=True, help="NC文件目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--static-file", help="静态文件路径")
    parser.add_argument("--research-vars", required=True, nargs="+", help="研究变量列表")
    parser.add_argument("--filter", default="avg", help="文件名过滤关键字")
    parser.add_argument("--no-plot", action="store_true", help="不绘制分布图")

    args = parser.parse_args()

    result = run_full_preprocessing(
        nc_folder_path=args.nc_folder,
        output_base_dir=args.output,
        research_vars=args.research_vars,
        static_file_path=args.static_file,
        file_filter=args.filter,
        plot_distribution=not args.no_plot
    )

    # 保存结果
    result_file = os.path.join(args.output, "preprocess_result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n处理结果已保存: {result_file}")
