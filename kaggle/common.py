"""Shared setup for the Kaggle kernels: install, fetch the code at a pinned commit, locate the data.

Kaggle uploads only the code_file of a kernel, so each kernel script inlines this file at push time
(see kaggle/push.py); edit it here.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/geminitt/monodist.git"
ULTRALYTICS = "ultralytics==8.4.163"
SRC = Path("/tmp/monodist")
WORK = Path("/kaggle/working")


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def setup(commit):
    sh(f"pip install -q {ULTRALYTICS}")
    if not SRC.exists():
        sh(f"git clone -q {REPO} {SRC}")
    sh(f"cd {SRC} && git fetch -q origin && git checkout -q {commit}")
    sys.path.insert(0, str(SRC))
    root = next(Path("/kaggle/input").glob("**/training/label_02")).parent
    print("KITTI training dir:", root, flush=True)
    import torch
    print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    return root


def log(name, **values):
    """Append one JSON line to /kaggle/working/<name>.jsonl (kept as kernel output)."""
    values["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(WORK / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(values) + "\n")
    print(name, values, flush=True)
