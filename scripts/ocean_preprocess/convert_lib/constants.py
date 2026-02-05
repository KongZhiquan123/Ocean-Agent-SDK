"""
constants.py - 常量定义

从 convert_npy.py 拆分

@changelog
  - 2026-02-05 kongzhiquan: 新增日期文件名格式常量
"""

# 静态变量编号前缀
# 00-09: 经度坐标
# 10-19: 纬度坐标
# 20-89: 其他静态变量
# 90-99: 掩码变量（排最后）

LON_VARS = ['lon_rho', 'lon_u', 'lon_v', 'lon_psi', 'longitude', 'lon']
LAT_VARS = ['lat_rho', 'lat_u', 'lat_v', 'lat_psi', 'latitude', 'lat']
MASK_VARS_DEFAULT = ['mask_rho', 'mask_u', 'mask_v', 'mask_psi']

# 时间坐标候选
TIME_COORD_CANDIDATES = ['ocean_time', 'time', 't', 'Time', 'TIME']

# NaN/Inf 检测采样参数
NAN_CHECK_SAMPLE_SIZE = 10000  # 采样点数
NAN_CHECK_RANDOM_SEED = 42     # 固定随机种子保证可复现

# 启发式掩码验证参数
HEURISTIC_SAMPLE_SIZE = 2000   # 陆地/海洋采样点数
LAND_THRESHOLD_ABS = 1e-12     # 陆地零值判定阈值
LAND_ZERO_RATIO_MIN = 0.90     # 陆地点零值比例下限
OCEAN_ZERO_RATIO_MAX = 0.90    # 海洋点零值比例上限

# 多线程默认参数
DEFAULT_WORKERS = 32

# 日期文件名格式（用于 use_date_filename 功能）
# 格式选项: "auto", "YYYYMMDD", "YYYYMMDDHH", "YYYYMMDDHHmm", "YYYY-MM-DD"
DEFAULT_DATE_FORMAT = "auto"

