# monodist: COCO-pretrained YOLO26n detections on the val and test sequences, on CPU (no GPU needed).
# Accuracy only; latency is measured on the GPU by the eval kernel.
MODE = "full"
COMMIT = "main"

# <common.py>

import os

root = setup(COMMIT)
os.chdir("/tmp")
from monodist import kitti  # noqa: E402

seqs = kitti.SPLIT["val"] + kitti.SPLIT["test"]
t = time.time()
sh(f"cd /tmp && PYTHONPATH={SRC} python -m monodist.detect --weights yolo26n.pt --coco --root {root} --out {WORK}/det/coco_1280 "
   f"--imgsz 1280 --seqs {' '.join(seqs)}" + (" --frames 10" if MODE == "smoke" else ""))
log("detect_cpu", step="detect", config="coco_1280", seconds=round(time.time() - t, 1))
