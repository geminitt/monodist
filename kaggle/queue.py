"""Run the cross-validation jobs on Kaggle, at most two at a time (the account's limit of batch GPU sessions),
and download each job's output when it finishes. Safe to stop and rerun: finished jobs are skipped.

    python kaggle/queue.py --commit <sha>                  # all folds x configurations a-d, plus e on fold 0
    python kaggle/queue.py --commit <sha> --jobs f0-d --mode smoke --prefix smoke-

A job is done when runs/kaggle/cv/<job>/ holds, for both heads, a readable detection file of every expected
sequence, and the kernel's log names the requested commit. The exit code is 1 if any job failed.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from monodist import kitti  # noqa: E402

OUT = HERE.parent / "runs" / "kaggle" / "cv"
USER = "spritker"
TERMINAL_BAD = ("ERROR", "CANCEL")
TIMEOUT = 300  # seconds for one kaggle CLI call: an operational limit, not a measured duration
# What Kaggle CLI 2.2.3 printed (exit 1) on 2026-10-13 for a kernel that does not exist or was deleted
NOT_FOUND = "Permission 'kernels.get' was denied"
DET_KEYS = {"frame", "box", "score", "cls", "image_hw"}


def kaggle(*args):
    """Run the kaggle CLI; None if it did not answer within TIMEOUT."""
    try:
        return subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None


def status(slug):
    """The kernel's status line, "" if it was never pushed, None if unknown (timeout or other CLI error).

    A job whose status is unknown must not be pushed: it may be running, and a push would start a second session.
    """
    r = kaggle("kernels", "status", slug)
    if r is None:
        return None
    text = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        return "" if NOT_FOUND in text else None
    lines = text.splitlines()
    return lines[-1] if lines else None


def valid_detections(f, seq):
    """A detection file that loads, has every field, one row per box and frame numbers inside the sequence."""
    try:
        with np.load(f) as d:
            if not DET_KEYS <= set(d.files):
                return False
            n = len(d["frame"])
            return (len(d["score"]) == len(d["cls"]) == n and d["box"].reshape(-1, 4).shape[0] == n
                    and bool(np.all((d["frame"] >= 0) & (d["frame"] < kitti.FRAMES[seq]))))
    except Exception:  # noqa: BLE001 -- a truncated or foreign file is simply not done
        return False


def logged_commit(folder):
    """The commit the kernel checked out, from its cv.jsonl log (None if absent)."""
    f = OUT / folder / "cv.jsonl"
    if not f.is_file():
        return None
    for line in f.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("step") == "setup":
            return rec.get("commit")
    return None


def done(folder, job, seqs, commit):
    name = job.replace("-", "_")          # f0-a -> f0_a, the run name inside the kernel
    det = OUT / folder / "det"
    return logged_commit(folder) == commit and all(valid_detections(det / f"{name}_{h}" / f"{s}.npz", s)
                                                   for h in ("nms", "e2e") for s in seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--jobs", nargs="*", default=[f"f{k}-{c}" for k in range(3) for c in "abcd"] + ["f0-e"])
    ap.add_argument("--mode", default="full")
    ap.add_argument("--prefix", default="", help="kernel name prefix, e.g. smoke- to keep smoke runs apart")
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()
    seqs = kitti.DEVELOPMENT[:2] if args.mode == "smoke" else kitti.DEVELOPMENT  # as in kaggle/cv/run.py
    failed = set()
    while True:
        todo = [j for j in args.jobs if not done(args.prefix + j, j, seqs, args.commit) and j not in failed]
        if not todo:
            break
        active = 0
        waiting = []
        for job in todo:
            slug = f"{USER}/monodist-cv-{args.prefix}{job}"
            s = status(slug)
            if s is None:
                print(f"{job}: status unknown (kaggle CLI timeout or error); not pushing it this round", flush=True)
                active += 1  # it may be running
            elif "COMPLETE" in s:
                (OUT / (args.prefix + job)).mkdir(parents=True, exist_ok=True)
                if kaggle("kernels", "output", slug, "-p", str(OUT / (args.prefix + job))) is None:
                    print(f"{job}: output download timed out; retrying next round", flush=True)
                    continue
                if not done(args.prefix + job, job, seqs, args.commit):
                    print(f"{job}: finished but its output is incomplete; see {OUT / (args.prefix + job)}", flush=True)
                    failed.add(job)
                else:
                    print(f"{job}: done", flush=True)
            elif any(b in s for b in TERMINAL_BAD):
                print(f"{job}: {s}", flush=True)
                failed.add(job)
            elif "QUEUED" in s or "RUNNING" in s:
                active += 1
            else:
                waiting.append(job)
        for job in waiting[:max(0, args.parallel - active)]:
            k, c = job[1], job[3]
            try:
                r = subprocess.run([sys.executable, str(HERE / "push.py"), "cv", "--mode", args.mode, "--commit", args.commit,
                                    "--job", args.prefix + job, "--set", f"FOLD={k}", f'CONFIG="{c}"'],
                                   capture_output=True, text=True, timeout=TIMEOUT)
                print(f"{job}: push -> {(r.stdout + r.stderr).strip().splitlines()[-1:]}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"{job}: push timed out; its status is checked next round", flush=True)
        time.sleep(60)
    print("finished; failed:", sorted(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
