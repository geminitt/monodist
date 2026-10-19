# Rebuilt quality pipeline: development calibration, then the saved ensemble/tracker/TTC.
# Attach the complete train output from COMMIT. No held-out test labels are evaluated here.
MODE = "smoke"
COMMIT = "main"
TRAIN_INPUT = None

# <common.py>

import traceback

root = setup(COMMIT)
from monodist import crossval  # noqa: E402

search = Path(TRAIN_INPUT) if TRAIN_INPUT else (SRC / "runs/kaggle/train" if LOCAL else Path("/kaggle/input"))


def input_for(job, head):
    matches = sorted(search.glob(f"**/det/{job}_{head}/run.json"))
    if not matches:
        raise FileNotFoundError(f"no {job}_{head} detections under {search}; attach a complete train output")
    return matches[0].parent.parent.parent


training = input_for("coco", "nms")
if not LOCAL:
    records = [json.loads(line) for line in (training / "train.jsonl").read_text().splitlines()]
    assert any(r.get("step") == "setup" and r.get("commit") == COMMIT for r in records), "train source commit differs"
    assert any(r.get("step") == "end" and r.get("failed") == [] for r in records), "training is incomplete"
    table, _ = crossval.select(root, training / "det", ["a", "b"])
    chosen = max(("a", "b"), key=lambda c: sum(table[k][c] for k in table) / len(table))
else:
    chosen = "b"  # local procedure check uses the available selected-model artifacts
log("eval", step="setup", commit=COMMIT, mode=MODE, final_configuration=chosen,
    protocol="sequential configurations on GPU 0; ensemble, thresholds, tracker and TTC included")

configs = [("coco_nms", "coco", "nms"), ("finetuned_nms", f"final_{chosen}", "nms"),
           ("finetuned_e2e", f"final_{chosen}", "e2e")]
failed = []
for name, job, head in configs:
    try:
        inputs = input_for(job, head)
        if not LOCAL and inputs != training:
            raise ValueError("all configurations must come from the same complete train output")
        det = inputs / "det" / f"{job}_{head}"
        weights = inputs / "runs" / job / "weights" / "best.pt" if job != "coco" else Path("/tmp/yolo26n.pt")
        if job == "coco" and not weights.exists():
            if LOCAL:
                import shutil
                shutil.copy(SRC / "weights" / "yolo26n.pt", weights)
            else:
                from ultralytics import YOLO
                os.chdir("/tmp")
                YOLO(str(weights))
        calibration = WORK / "calibration" / name
        sh(f"PYTHONPATH={SRC} {sys.executable} -m monodist.calibrate --root {root} --det {det} --out {calibration}",
           timeout=remaining())
        # A controlled single-pipeline comparison needs the same GPU and no concurrent CPU decode workload.
        # This timing protocol needs the owner's explicit exception to the usual two-busy-GPUs rule.
        frames = 3 if LOCAL else (20 if MODE == "smoke" else 300)
        seq = "0012" if LOCAL else "0019"
        rounds, warmup = (1, 1) if LOCAL else ((1, 5) if MODE == "smoke" else (3, 30))
        for pipelined in (False, True):
            label = name + ("_pipelined" if pipelined else "")
            cmd = (f"CUDA_VISIBLE_DEVICES=0 PYTHONPATH={SRC} {sys.executable} -m monodist.latency --weights {weights} "
                   f"--distance {calibration}/distance.pt --root {root} --seq {seq} --frames {frames} --rounds {rounds} "
                   f"--warmup {warmup} --imgsz 1280 --out {WORK}/latency/{label}.json"
                   + (" --nms" if head == "nms" else "") + (" --pipelined" if pipelined else ""))
            sh(cmd, timeout=remaining())
            log("eval", step="latency", configuration=label)
    except Exception:
        failed.append(name)
        log("eval", step="failed", configuration=name, error=traceback.format_exc())
log("eval", step="end", failed=failed, seconds=time.time() - START)
if failed:
    raise SystemExit(f"failed configurations: {failed}")
