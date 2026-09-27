# monodist: detections for every configuration on all 21 sequences, then per-frame latency (GPU T4).
# Needs the output of the monodist-finetune kernel (best.pt). MODE = "smoke" runs a few frames of everything.
MODE = "smoke"
COMMIT = "main"
ONLY = None  # e.g. ["ft_1280_trt16"]: run only these configurations (and their pipelined variants)

# <common.py>

import os
import traceback

root = setup(COMMIT)
os.chdir("/tmp")  # Ultralytics downloads yolo26n.pt and writes exports into the working directory
from ultralytics import YOLO  # noqa: E402

from monodist import kitti  # noqa: E402

smoke = MODE == "smoke"
import shutil  # noqa: E402
ft = "/tmp/ft.pt"  # exports are written next to the weights, and /kaggle/input is read-only
shutil.copy(next(Path("/kaggle/input").glob("**/finetune/weights/best.pt")), ft)
coco = "yolo26n.pt"
YOLO(coco)  # download once
print("fine-tuned weights:", ft, flush=True)

# Fixed-shape TensorRT engines. KITTI frames (~1242 x 375) letterboxed to a long side of 1280 become 1280 x 387,
# padded to a multiple of 32: 1280 x 416. At 640: 640 x 193 -> 640 x 224.
engines = {}
for name, shape in (("ft_1280_trt16", [416, 1280]), ("ft_640_trt16", [224, 640])):
    try:
        if ONLY and name not in ONLY:
            continue
        path = Path(YOLO(ft).export(format="engine", quantize=16, imgsz=shape, nms=False, batch=1, device=0))
        path = path.rename(path.with_name(f"{name}.engine"))  # every export writes ft.engine: keep each one
        engines[name] = (path, f"{shape[0]},{shape[1]}")
        log("eval", step="export", config=name, path=str(path))
    except Exception:
        log("eval", step="export", config=name, error=traceback.format_exc()[-2000:])

configs = [("coco_1280", coco, "1280", False, True), ("ft_1280", ft, "1280", False, False),
           ("ft_1280_fp16", ft, "1280", True, False), ("ft_640", ft, "640", False, False)]
configs += [(n, p, s, False, False) for n, (p, s) in engines.items()]
if ONLY:
    configs = [c for c in configs if c[0] in ONLY]

seqs = kitti.SEQUENCES  # train sequences too: the distance MLP trains on each detector's own train-split boxes
if smoke:
    seqs = ["0003", "0014"]
for name, weights, imgsz, half, is_coco in configs:
    cmd = (f"cd /tmp && PYTHONPATH={SRC} python -m monodist.detect --weights {weights} --root {root} --out {WORK}/det/{name} "
           f"--imgsz {imgsz} --seqs {' '.join(seqs)}" + (" --half" if half else "") + (" --coco" if is_coco else "")
           + (" --frames 20" if smoke else "") + (" --batch 1" if str(weights).endswith(".engine") else ""))
    t = time.time()
    try:
        sh(cmd)
        log("eval", step="detect", config=name, seconds=round(time.time() - t, 1))
    except Exception:
        log("eval", step="detect", config=name, error=traceback.format_exc()[-2000:])

lat = "--frames 30 --rounds 1 --warmup 5" if smoke else "--frames 300 --rounds 3 --warmup 30"
runs = [(n, w, s, h, False) for n, w, s, h, _ in configs]
runs += [(n + "_pipelined", w, s, h, True) for n, w, s, h, _ in configs if n in ("ft_1280", "ft_1280_trt16")]
for name, weights, imgsz, half, pipelined in runs:
    cmd = (f"cd /tmp && PYTHONPATH={SRC} python -m monodist.latency --weights {weights} --root {root} --out {WORK}/latency/{name}.json "
           f"--imgsz {imgsz} {lat}" + (" --half" if half else "") + (" --pipelined" if pipelined else ""))
    try:
        sh(cmd)
    except Exception:
        log("eval", step="latency", config=name, error=traceback.format_exc()[-2000:])
log("eval", step="hardware", cpu_count=os.cpu_count(), affinity=len(os.sched_getaffinity(0)))
sh("lscpu | grep 'Model name'; nvidia-smi --query-gpu=name,clocks.max.sm --format=csv")
