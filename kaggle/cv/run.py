# monodist: one cross-validation run = one fold x one fine-tuning configuration (GPU T4).
# Fine-tunes on the fold's train sequences (epoch chosen on its val sequences), then detects on all 21
# sequences with both YOLO26 heads. The kernel id carries the fold and configuration (kaggle/push.py --job).
MODE = "full"
COMMIT = "main"
FOLD = 0
CONFIG = "a"

# <common.py>

import os
import traceback

root = setup(COMMIT)
os.chdir("/tmp")
from monodist import kitti, yolo_data  # noqa: E402
from monodist.finetune import CONFIGS  # noqa: E402

smoke = MODE == "smoke"
name = f"f{FOLD}_{CONFIG}"
cfg = CONFIGS[CONFIG]
classes = kitti.CLASSES + (kitti.EXTRA_CLASSES if cfg.get("extra_classes") else ())
t = time.time()
yaml = yolo_data.build(root, "/tmp/yolo", split=kitti.FOLDS[FOLD], splits=("train", "val"), classes=classes,
                       mask_dontcare=cfg.get("mask_dontcare", False), frames=20 if smoke else None)
log("cv", step="yolo_data", job=name, classes=list(classes), seconds=round(time.time() - t, 1))

args = f"--data {yaml} --project {WORK}/runs --name {name} --config {CONFIG}"
args += " --epochs 1 --cache False" if smoke else ""
t = time.time()
sh(f"cd /tmp && PYTHONPATH={SRC} python -m monodist.finetune {args}")
log("cv", step="finetune", job=name, seconds=round(time.time() - t, 1))

best = WORK / "runs" / name / "weights" / "best.pt"
(WORK / "runs" / name / "weights" / "last.pt").unlink(missing_ok=True)  # only best.pt is used; saves output space
seqs = kitti.SEQUENCES[:2] if smoke else kitti.SEQUENCES
for head, flag in (("nms", " --nms"), ("e2e", "")):
    t = time.time()
    try:
        sh(f"cd /tmp && PYTHONPATH={SRC} python -m monodist.detect --weights {best} --root {root} "
           f"--out {WORK}/det/{name}_{head} --seqs {' '.join(seqs)}{flag}" + (" --frames 20" if smoke else ""))
        log("cv", step="detect", job=name, head=head, seconds=round(time.time() - t, 1))
    except Exception:
        log("cv", step="detect", job=name, head=head, error=traceback.format_exc()[-2000:])
