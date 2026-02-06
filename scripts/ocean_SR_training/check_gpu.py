"""
check_gpu.py - 查询当前可用的 GPU 信息

输出 JSON 格式的 GPU 信息，供 TypeScript 工具调用。
"""

import json
import subprocess
import shutil


def check_gpu():
    """查询可用 GPU 信息"""
    result = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
    }

    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["gpu_count"] = torch.cuda.device_count()

        for i in range(result["gpu_count"]):
            props = torch.cuda.get_device_properties(i)
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            result["gpus"].append({
                "id": i,
                "name": props.name,
                "total_memory_gb": round(total_mem / 1024**3, 2),
                "free_memory_gb": round(free_mem / 1024**3, 2),
                "used_memory_gb": round((total_mem - free_mem) / 1024**3, 2),
            })
    except ImportError:
        # torch 不可用时，尝试用 nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True
                )
                lines = [l.strip() for l in out.strip().split("\n") if l.strip()]
                result["gpu_count"] = len(lines)
                result["cuda_available"] = len(lines) > 0
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    result["gpus"].append({
                        "id": int(parts[0]),
                        "name": parts[1],
                        "total_memory_gb": round(float(parts[2]) / 1024, 2),
                        "free_memory_gb": round(float(parts[3]) / 1024, 2),
                        "used_memory_gb": round(float(parts[4]) / 1024, 2),
                    })
            except Exception as e:
                result["error"] = f"nvidia-smi failed: {str(e)}"
        else:
            result["error"] = "Neither torch nor nvidia-smi available"

    return result


if __name__ == "__main__":
    info = check_gpu()
    print(json.dumps(info, ensure_ascii=False, indent=2))
