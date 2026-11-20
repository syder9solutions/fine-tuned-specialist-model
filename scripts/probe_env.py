"""Environment probe: GPU, Python, key ML packages, network reachability."""
import importlib.util
import json
import platform
import shutil
import subprocess
import sys


def has(mod: str) -> str:
    spec = importlib.util.find_spec(mod)
    if spec is None:
        return "MISSING"
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "present")
    except Exception as exc:  # present but broken
        return f"present (import error: {exc})"


probe = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": __import__("os").cpu_count(),
    "mem_total_gb": round(__import__("os").sysconf("SC_PAGE_SIZE") * __import__("os").sysconf("SC_PHYS_PAGES") / 2**30, 1)
    if hasattr(__import__("os"), "sysconf")
    else None,
    "nvidia_smi": shutil.which("nvidia-smi"),
    "packages": {
        "torch": has("torch"),
        "transformers": has("transformers"),
        "peft": has("peft"),
        "trl": has("trl"),
        "datasets": has("datasets"),
        "bitsandbytes": has("bitsandbytes"),
        "unsloth": has("unsloth"),
        "sklearn": has("sklearn"),
        "numpy": has("numpy"),
    },
}

if probe["nvidia_smi"]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        probe["gpus"] = out
    except Exception as exc:
        probe["gpus"] = f"error: {exc}"
else:
    probe["gpus"] = "none"

print(json.dumps(probe, indent=2))
