"""Inline common.py into a kernel script, pin the commit, and push it with the kaggle CLI.

    python kaggle/push.py finetune --mode smoke --commit <sha>
"""
import argparse
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("kernel")
ap.add_argument("--mode", default=None)
ap.add_argument("--commit", default=None, help="defaults to the current HEAD, which must be pushed")
ap.add_argument("--set", nargs="*", default=[], help="NAME=VALUE overrides of top-level constants")
args = ap.parse_args()

commit = args.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
remote = subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True).stdout
assert remote.strip(), f"commit {commit} is not on the remote yet: push it first"

src = (HERE / args.kernel / "run.py").read_text()
common = (HERE / "common.py").read_text()
src = src.replace("# <common.py>", common)
src = re.sub(r'^COMMIT = .*$', f'COMMIT = "{commit}"', src, flags=re.M)
if args.mode:
    src = re.sub(r'^MODE = .*$', f'MODE = "{args.mode}"', src, flags=re.M)
for kv in args.set:
    k, v = kv.split("=", 1)
    src, n = re.subn(rf'^{k} = .*$', f"{k} = {v}", src, flags=re.M)
    assert n == 1, f"no top-level constant {k}"
build = HERE / args.kernel / "build"
build.mkdir(exist_ok=True)
(build / "run.py").write_text(src)
(build / "kernel-metadata.json").write_text((HERE / args.kernel / "kernel-metadata.json").read_text())
subprocess.run(["kaggle", "kernels", "push", "-p", str(build)], check=True)
