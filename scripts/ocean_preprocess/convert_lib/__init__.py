"""
convert_lib - NC 转 NPY 转换库

从 convert_npy.py 拆分的模块化库

@author leizheng
@contributors kongzhiquan
@date 2026-02-05
@version 3.2.0

模块结构:
- constants.py: 常量定义
- encoder.py: JSON 编码器
- crop.py: 裁剪相关函数
- check.py: 数据检测函数
- mask.py: 掩码相关函数
- converters.py: 变量转换函数
- validation.py: 验证规则
- time_utils.py: 时间处理函数（新增）
- core.py: 核心转换函数

@changelog
  - 2026-02-05 kongzhiquan: v3.2.0 新增日期文件名功能
    - 新增 time_utils.py 模块
    - 支持从 NC 文件提取时间戳作为文件名
  - 2026-02-05 kongzhiquan: v3.1.1 初始版本，从 convert_npy.py 拆分
"""

# 核心函数
from .core import convert_npy

# JSON 编码器
from .encoder import NumpyEncoder

# 常量
from .constants import (
    LON_VARS,
    LAT_VARS,
    MASK_VARS_DEFAULT,
    TIME_COORD_CANDIDATES,
    NAN_CHECK_SAMPLE_SIZE,
    NAN_CHECK_RANDOM_SEED,
    HEURISTIC_SAMPLE_SIZE,
    LAND_THRESHOLD_ABS,
    LAND_ZERO_RATIO_MIN,
    OCEAN_ZERO_RATIO_MAX,
    DEFAULT_WORKERS,
    DEFAULT_DATE_FORMAT
)

# 裁剪函数
from .crop import (
    parse_slice_str,
    crop_spatial,
    validate_crop_divisible,
    get_cropped_shape,
    compute_region_crop_indices,
    adjust_crop_for_scale,
    load_coordinate_arrays
)

# 检测函数
from .check import (
    is_object_dtype,
    check_nan_inf_sampling,
    get_spatial_shape,
    verify_coordinate_range,
    get_static_var_prefix,
    find_time_coord
)

# 掩码函数
from .mask import (
    derive_mask,
    derive_staggered_mask,
    heuristic_mask_check
)

# 转换函数
from .converters import (
    convert_dynamic_vars,
    convert_static_vars
)

# 验证函数
from .validation import (
    validate_rule1,
    validate_rule2,
    validate_rule3
)

# 时间处理函数
from .time_utils import (
    extract_timestamps_from_files,
    detect_date_format,
    generate_date_filenames,
    validate_time_monotonic,
    create_time_mapping,
    DATE_FORMATS
)

__all__ = [
    # 核心
    'convert_npy',
    'NumpyEncoder',

    # 常量
    'LON_VARS',
    'LAT_VARS',
    'MASK_VARS_DEFAULT',
    'TIME_COORD_CANDIDATES',
    'NAN_CHECK_SAMPLE_SIZE',
    'NAN_CHECK_RANDOM_SEED',
    'HEURISTIC_SAMPLE_SIZE',
    'LAND_THRESHOLD_ABS',
    'LAND_ZERO_RATIO_MIN',
    'OCEAN_ZERO_RATIO_MAX',
    'DEFAULT_WORKERS',
    'DEFAULT_DATE_FORMAT',

    # 裁剪
    'parse_slice_str',
    'crop_spatial',
    'validate_crop_divisible',
    'get_cropped_shape',
    'compute_region_crop_indices',
    'adjust_crop_for_scale',
    'load_coordinate_arrays',

    # 检测
    'is_object_dtype',
    'check_nan_inf_sampling',
    'get_spatial_shape',
    'verify_coordinate_range',
    'get_static_var_prefix',
    'find_time_coord',

    # 掩码
    'derive_mask',
    'derive_staggered_mask',
    'heuristic_mask_check',

    # 转换
    'convert_dynamic_vars',
    'convert_static_vars',

    # 验证
    'validate_rule1',
    'validate_rule2',
    'validate_rule3',

    # 时间处理
    'extract_timestamps_from_files',
    'detect_date_format',
    'generate_date_filenames',
    'validate_time_monotonic',
    'create_time_mapping',
    'DATE_FORMATS',
]
