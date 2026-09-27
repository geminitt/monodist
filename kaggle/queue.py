"""Run the cross-validation jobs on Kaggle, at most two at a time (the account's limit of batch GPU sessions),
and download each job's output when it finishes. Safe to stop and rerun: finished jobs are skipped.

    python kaggle/queue.py --commit <sha>                  # all folds x configurations a-d, plus e on fold 0
    python kaggle/queue.py --commit <sha> --jobs f0-d --mode smoke --prefix smoke-

A job is done when runs/kaggle/cv/<job>/det/ holds both heads' detections for all 21 sequences.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "runs" / "kaggle" / "cv"
USER = "spritker"
TERMINAL_BAD = ("ERROR", "CANCEL")


def kaggle(*args):
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


def status(slug):
    r = kaggle("kernels", "status", slug)
    text = (r.stdout + r.stderr).strip().splitlines()
    return text[-1] if text else ""


def done(folder, job, n_seqs):
    name = job.replace("-", "_")          # f0-a -> f0_a, the run name inside the kernel
    det = OUT / folder / "det"
    return all(len(list((det / f"{name}_{h}").glob("*.npz"))) == n_seqs for h in ("nms", "e2e"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--jobs", nargs="*", default=[f"f{k}-{c}" for k in range(3) for c in "abcd"] + ["f0-e"])
    ap.add_argument("--mode", default="full")
    ap.add_argument("--prefix", default="", help="kernel name prefix, e.g. smoke- to keep smoke runs apart")
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()
    n_seqs = 2 if args.mode == "smoke" else 21
    failed = set()
    while True:
        todo = [j for j in args.jobs if not done(args.prefix + j, j, n_seqs) and j not in failed]
        if not todo:
            break
        active = 0
        waiting = []
        for job in todo:
            slug = f"{USER}/monodist-cv-{args.prefix}{job}"
            s = status(slug)
            if "COMPLETE" in s:
                (OUT / (args.prefix + job)).mkdir(parents=True, exist_ok=True)
                kaggle("kernels", "output", slug, "-p", str(OUT / (args.prefix + job)))
                if not done(args.prefix + job, job, n_seqs):
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
            r = subprocess.run([sys.executable, str(HERE / "push.py"), "cv", "--mode", args.mode, "--commit", args.commit,
                                "--job", args.prefix + job, "--set", f"FOLD={k}", f'CONFIG="{c}"'],
                               capture_output=True, text=True)
            print(f"{job}: push -> {(r.stdout + r.stderr).strip().splitlines()[-1:]}", flush=True)
        time.sleep(60)
    print("finished; failed:", sorted(failed), flush=True)


if __name__ == "__main__":
    main()
