"""
check_gpu.py - 查询当前可用的 GPU 信息

输出 JSON 格式的 GPU 信息，供 TypeScript 工具调用。

@author kongzhiquan
@date 2026-02-09
@version 1.2.0

@changelog
  - 2026-02-10 Leizheng: v1.2.0 nvidia-smi 增加 timeout=15s 防挂起
  - 2026-02-09 kongzhiquan: v1.1.0 增加单卡 RuntimeError 容错，回退 nvidia-smi 查询
"""

import json
import shutil
import subprocess


def _query_gpu_by_nvidia_smi():
    """优先使用 nvidia-smi 查询，避免触发 CUDA 上下文分配导致的 OOM。"""
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found")

    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )

    lines = [line.strip() for line in out.strip().split("\n") if line.strip()]
    result = {
        "cuda_available": len(lines) > 0,
        "gpu_count": len(lines),
        "gpus": [],
    }

    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        total_mem = float(parts[2])
        free_mem = float(parts[3])
        used_mem = float(parts[4])
        result["gpus"].append(
            {
                "id": int(parts[0]),
                "name": parts[1],
                "total_memory_gb": round(total_mem / 1024, 2),
                "free_memory_gb": round(free_mem / 1024, 2),
                "used_memory_gb": round(used_mem / 1024, 2),
            }
        )

    return result


def _query_gpu_by_torch():
    """当 nvidia-smi 不可用时，退化使用 torch.cuda。"""
    import torch

    result = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }

    for i in range(result["gpu_count"]):
        props = torch.cuda.get_device_properties(i)
        total_gb = round(props.total_memory / 1024**3, 2)
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            total_gb = round(total_mem / 1024**3, 2)
            free_gb = round(free_mem / 1024**3, 2)
            used_gb = round((total_mem - free_mem) / 1024**3, 2)
        except Exception as e:
            # 某张卡 mem_get_info 失败时，不让整个工具失败；用保守值继续。
            free_gb = 0.0
            used_gb = total_gb
            result["error"] = f"torch.cuda.mem_get_info partial failure: {type(e).__name__}"

        result["gpus"].append(
            {
                "id": i,
                "name": props.name,
                "total_memory_gb": total_gb,
                "free_memory_gb": free_gb,
                "used_memory_gb": used_gb,
            }
        )

    return result


def check_gpu():
    """查询可用 GPU 信息。"""
    # 先走 nvidia-smi（更稳，不依赖 CUDA 上下文）；失败再走 torch。
    smi_error = None
    try:
        return _query_gpu_by_nvidia_smi()
    except Exception as e:
        smi_error = f"nvidia-smi failed: {e}"

    try:
        result = _query_gpu_by_torch()
        if smi_error:
            result["error"] = smi_error
        return result
    except ImportError:
        return {
            "cuda_available": False,
            "gpu_count": 0,
            "gpus": [],
            "error": smi_error or "Neither torch nor nvidia-smi available",
        }
    except Exception as e:
        return {
            "cuda_available": False,
            "gpu_count": 0,
            "gpus": [],
            "error": f"{smi_error}; torch fallback failed: {type(e).__name__}: {e}" if smi_error else f"torch fallback failed: {type(e).__name__}: {e}",
        }


if __name__ == "__main__":
    info = check_gpu()
    print(json.dumps(info, ensure_ascii=False, indent=2))
