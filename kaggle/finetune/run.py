# monodist: fine-tune YOLO26n on the KITTI train sequences (GPU T4).
# MODE = "smoke": one short epoch on 10% of the images, to time an epoch. MODE = "full": the real run.
MODE = "smoke"
COMMIT = "main"
EPOCHS = 30

# <common.py>

import time

root = setup(COMMIT)
from monodist import yolo_data  # noqa: E402

t = time.time()
yaml = yolo_data.build(root, "/tmp/yolo", splits=("train", "val"))
log("finetune", step="yolo_data", seconds=round(time.time() - t, 1))

args = f"--data {yaml} --project {WORK}/runs --name finetune --imgsz 1280 --batch 16"
if MODE == "smoke":
    args += " --epochs 1 --fraction 0.1 --cache False"
else:
    args += f" --epochs {EPOCHS}"
t = time.time()
sh(f"cd {SRC} && python -m monodist.finetune {args}")
log("finetune", step="train", mode=MODE, seconds=round(time.time() - t, 1))
