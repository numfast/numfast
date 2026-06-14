from importlib.metadata import version, PackageNotFoundError
import subprocess
import os

def _get_version() -> str:
    try:
        return version("numfast")
    except PackageNotFoundError:
        return "0.0.0-dev"

def _get_status() -> dict:
    return {
        "name": "numfast",
        "version": _get_version(),
        "extensions": [
            "_main",
            "Series",
            "Tables",
            "Memory",
            "Mods",
            "Proxy",
        ],
    }

def _check_cuda() -> dict:
    result = {"available": False, "device": None, "memory": None}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(",")
            result["device"] = parts[0].strip()
            result["memory"] = parts[1].strip() if len(parts) > 1 else None
            result["available"] = True
    except Exception:
        try:
            import cupy
            result["available"] = cupy.cuda.is_available()
            if result["available"]:
                prop = cupy.cuda.runtime.getDeviceProperties(0)
                result["device"] = prop["name"].decode() if isinstance(prop["name"], bytes) else prop["name"]
                result["memory"] = f"{prop['totalGlobalMem'] // 1048576} MiB"
        except Exception:
            pass
    return result

def _check_rocm() -> dict:
    result = {"available": False, "device": None, "reason": None}
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            result["available"] = True
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith("Name:"):
                    result["device"] = line.split(":", 1)[1].strip()
                    break
        else:
            result["reason"] = "rocminfo returned error"
    except FileNotFoundError:
        result["reason"] = "rocminfo not found"
    except Exception as e:
        result["reason"] = str(e)
    if not result["available"]:
        try:
            import torch
            if torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip:
                result["available"] = True
                result["device"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return result

def _check_webgpu() -> dict:
    result = {"available": False, "adapter": None, "backend_type": None, "device": None}
    try:
        import wgpu
        import os
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            adapter = wgpu.gpu.request_adapter_sync()
            info = adapter.info
            result["available"] = True
            result["adapter"] = info.get("device", "")
            result["backend_type"] = info.get("adapter_type", "")
            result["device"] = info.get("device", "")
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    except Exception:
        pass
    return result

def _check_cpu() -> dict:
    return {"cores": os.cpu_count() or 0}

def _recommend_backend(cuda: dict, rocm: dict, webgpu: dict) -> str:
    if cuda["available"]:
        return "CUDA"
    if rocm["available"]:
        return "ROCm"
    if webgpu["available"]:
        return "WebGPU"
    return "CPU"
