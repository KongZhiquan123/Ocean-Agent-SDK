"""
encoder.py - 自定义 JSON Encoder

从 convert_npy.py 拆分
"""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    自定义 JSON Encoder，用于处理 numpy 类型

    解决 "Object of type int64 is not JSON serializable" 错误
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
