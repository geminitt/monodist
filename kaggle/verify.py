"""Download the output of the train kernel and check all of it before anything uses it.

    python kaggle/verify.py --commit <sha> --mode full --download     # status, download, check
    python kaggle/verify.py --commit <sha> --mode full                # check an output already downloaded

Checks: the kernel's log names the requested commit and mode and ends without failed jobs; every job has, for
both heads, a readable detection file of every expected sequence with the expected settings; the weights each
detection run names (SHA-256 in run.json) are the best.pt of that job. The exit code is 1 on any failure.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from monodist import kitti  # noqa: E402

OUT = HERE.parent / "runs" / "kaggle" / "train"
SLUG = "spritker/monodist-train"
TIMEOUT = 300  # seconds for one kaggle CLI call: an operational limit, not a measured duration
# What Kaggle CLI 2.2.3 printed (exit 1) on 2026-10-13 for a kernel that does not exist or was deleted
NOT_FOUND = "Permission 'kernels.get' was denied"
DET_KEYS = {"frame", "box", "score", "cls", "image_hw"}
CONFIGS = ("a", "b")


def kaggle(*args):
    """Run the kaggle CLI; None if it did not answer within TIMEOUT."""
    try:
        return subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None


def status(slug):
    """The kernel's status line, "" if it was never pushed, None if unknown (timeout or other CLI error)."""
    r = kaggle("kernels", "status", slug)
    if r is None:
        return None
    text = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        return "" if NOT_FOUND in text else None
    lines = text.splitlines()
    return lines[-1] if lines else None


def expected_jobs():
    """Job name -> sequences it detects, as kaggle/train/run.py defines them."""
    jobs = {f"final_{c}": kitti.SEQUENCES for c in CONFIGS}
    jobs.update({f"f{k}_{c}": kitti.DEVELOPMENT for c in CONFIGS for k in range(3)})
    jobs["coco"] = kitti.SEQUENCES
    return jobs


def valid_detections(f, seq):
    """A detection file that loads, has every field, one row per box and frame numbers inside the sequence."""
    try:
        with np.load(f) as d:
            if not DET_KEYS <= set(d.files):
                return False
            n = len(d["frame"])
            return (len(d["score"]) == len(d["cls"]) == n and d["box"].reshape(-1, 4).shape[0] == n
                    and bool(np.all((d["frame"] >= 0) & (d["frame"] < kitti.FRAMES[seq]))))
    except Exception:  # noqa: BLE001 -- a truncated or foreign file is a failure, reported by the caller
        return False


def read_log(folder):
    rows = []
    f = Path(folder) / "train.jsonl"
    for line in f.read_text().splitlines() if f.is_file() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"step": "unreadable line"})
    return rows


def check(folder, commit, mode, jobs=None):
    """A list of problems (empty when the output is complete and consistent)."""
    folder = Path(folder)
    problems = []
    log = read_log(folder)
    setup = [r for r in log if r.get("step") == "setup"]
    if not setup or setup[0].get("commit") != commit or setup[0].get("mode") != mode:
        problems.append(f"log: setup {setup[:1]} is not commit {commit}, mode {mode}")
    end = [r for r in log if r.get("step") == "end"]
    if not end:
        problems.append("log: no end record (the kernel stopped early)")
    elif end[-1].get("failed"):
        problems.append(f"log: failed jobs {end[-1]['failed']}")
    frames = 20 if mode == "smoke" else None
    for job, seqs in expected_jobs().items():
        if jobs and job not in jobs:
            continue
        best = folder / "runs" / job / "weights" / "best.pt"
        sha = hashlib.sha256(best.read_bytes()).hexdigest() if best.is_file() else None
        if job != "coco" and sha is None:
            problems.append(f"{job}: no best.pt")
        for head in ("nms", "e2e"):
            d = folder / "det" / f"{job}_{head}"
            bad = [s for s in seqs if not valid_detections(d / f"{s}.npz", s)]
            if bad:
                problems.append(f"{job}_{head}: missing or invalid detections for {bad}")
            run = d / "run.json"
            if not run.is_file():
                problems.append(f"{job}_{head}: no run.json")
                continue
            sig = json.loads(run.read_text())
            if sig.get("frames") != frames or sig.get("nms") != (head == "nms"):
                problems.append(f"{job}_{head}: settings {sig} do not match mode {mode} and head {head}")
            if job != "coco" and sig["weights"]["sha256"] != sha:
                problems.append(f"{job}_{head}: detections were made with other weights than runs/{job}/best.pt")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--mode", default="full", choices=("smoke", "full"))
    ap.add_argument("--dir", default=None, help="downloaded output (default runs/kaggle/train/<mode>)")
    ap.add_argument("--jobs", nargs="*", default=None)
    ap.add_argument("--download", action="store_true", help="check the kernel's status and download its output first")
    args = ap.parse_args()
    folder = Path(args.dir) if args.dir else OUT / args.mode
    if args.download:
        s = status(SLUG)
        if s is None or "COMPLETE" not in s and "ERROR" not in s:
            print(f"{SLUG}: status {s!r}; not downloading")
            return 1
        folder.mkdir(parents=True, exist_ok=True)
        if kaggle("kernels", "output", SLUG, "-p", str(folder)) is None:
            print("download timed out")
            return 1
    problems = check(folder, args.commit, args.mode, args.jobs)
    for p in problems:
        print("PROBLEM", p)
    print(f"{folder}: {'complete' if not problems else f'{len(problems)} problems'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
