"""Shared setup for the Kaggle kernels: install, fetch the code at a pinned commit, locate the data.

Kaggle uploads only the code_file of a kernel, so each kernel script inlines this file at push time
(see kaggle/push.py); edit it here.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/geminitt/monodist.git"
ULTRALYTICS = "ultralytics==8.4.163"
SRC = Path("/tmp/monodist")
WORK = Path("/kaggle/working")
START = time.time()
DEADLINE = 11 * 3600  # seconds from kernel start: stop work before Kaggle's 12-hour session limit


def remaining():
    """Seconds left before DEADLINE (at least 1, so a command started after it times out at once)."""
    return max(1.0, DEADLINE - (time.time() - START))


def sh(cmd, timeout=None):
    """Run a shell command; raise CalledProcessError on failure, TimeoutExpired after timeout seconds.

    The command runs in its own process group, so a timeout kills the whole command (e.g. the python under
    "cd ... && python ..."), not only the shell.
    """
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, start_new_session=True)
    try:
        code = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        print(f"timed out after {timeout:.0f} s: {cmd}", flush=True)
        raise
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def setup(commit):
    sh(f"pip install -q {ULTRALYTICS}", timeout=min(1200, remaining()))
    if not SRC.exists():
        sh(f"git clone -q {REPO} {SRC}", timeout=min(600, remaining()))
    sh(f"cd {SRC} && git fetch -q origin && git checkout -q {commit}", timeout=min(600, remaining()))
    sys.path.insert(0, str(SRC))
    root = next(Path("/kaggle/input").glob("**/training/label_02")).parent
    print("KITTI training dir:", root, flush=True)
    import torch
    print("torch", torch.__version__, "| GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    return root


def fetch(url, dest, tries=5):
    """Download url to dest, retrying with exponential backoff (1, 2, 4, 8 s); raise after the last try."""
    import urllib.request
    for k in range(tries):
        try:
            tmp = Path(str(dest) + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return Path(dest)
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print(f"download failed ({e}); retrying in {2 ** k} s", flush=True)
            time.sleep(2 ** k)


def log(name, **values):
    """Append one JSON line to /kaggle/working/<name>.jsonl (kept as kernel output)."""
    values["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(WORK / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(values) + "\n")
    print(name, values, flush=True)
