# monodist: every fine-tuning and detection run of the rebuild in one session, one job per T4 at a time.
# Jobs: f<k>_<cfg> fine-tunes on fold k (kitti.FOLDS) and detects the 16 development sequences; final_<cfg>
# fine-tunes on kitti.SPLIT and detects all 21; coco detects all 21 with the COCO weights. Both heads each.
# MODE = "smoke": 20 frames per sequence and one epoch, every job type, both GPUs busy. MODE = "full": the real run.
MODE = "smoke"
COMMIT = "main"
JOBS = None  # e.g. ["f0_a", "coco"]; None runs every job
CONFIGS = ["a", "b"]  # keys of monodist.finetune.CONFIGS

# <common.py>

import queue
import shutil
import psutil
import threading
import traceback

root = setup(COMMIT)
log("train", step="setup", commit=COMMIT, mode=MODE)  # the verification of the outputs checks this commit
from monodist import finetune, kitti, yolo_data  # noqa: E402

smoke = MODE == "smoke"
if LOCAL:  # only one sequence has images on the local machine
    folds, split, dev, every = [{"train": ["0012"], "val": ["0012"]}] * 3, {"train": ["0012"], "val": ["0012"]}, ["0012"], ["0012"]
else:
    folds, split, dev, every = kitti.FOLDS, kitti.SPLIT, kitti.DEVELOPMENT, kitti.SEQUENCES

jobs = [(f"final_{c}", split, every, c) for c in CONFIGS]  # the longest first, so the two queues end together
jobs += [(f"f{k}_{c}", folds[k], dev, c) for c in CONFIGS for k in range(3)]
jobs += [("coco", None, every, None)]
if JOBS:
    jobs = [j for j in jobs if j[0] in JOBS]

import torch  # noqa: E402

devices = [str(i) for i in range(torch.cuda.device_count())] or ["cpu"]
mem = psutil.virtual_memory()
if LOCAL:
    devices = devices[:1] * 2  # two queues on the one local GPU, as on Kaggle's two
MASKED = Path("/tmp/masked_dontcare")  # one set of masked training images, shared by every masked job
masked_seqs = sorted({q for _, part, _, cfg in jobs if cfg and finetune.CONFIGS[cfg].get("dontcare_masked")
                      for p in ("train", "val") for q in part[p]})
# Masked PNG copies (compress_level 1) took 1.06 MiB per frame on sequence 0012 (2026-10-21); 2x is the margin.
masked_need = 2 * 1.06 * 2**20 * sum(20 if smoke else kitti.FRAMES[q] for q in masked_seqs)
disk_free = shutil.disk_usage("/tmp").free
log("train", step="preflight", devices=devices, jobs=[j[0] for j in jobs],
    ram_total_gib=round(mem.total / 2**30, 1), ram_available_gib=round(mem.available / 2**30, 1),
    tmp_free_gib=round(disk_free / 2**30, 1), masked_need_gib=round(masked_need / 2**30, 1),
    gpu_mem_used_mib=[torch.cuda.mem_get_info(i)[1] // 2**20 - torch.cuda.mem_get_info(i)[0] // 2**20
                      for i in range(torch.cuda.device_count())])
coco_pt = Path("/tmp/yolo26n.pt")
if not coco_pt.exists():
    if LOCAL:
        shutil.copy(SRC / "weights" / "yolo26n.pt", coco_pt)
    else:
        from ultralytics import YOLO
        os.chdir("/tmp")
        YOLO("yolo26n.pt")  # one download before the parallel jobs
if masked_seqs:
    if disk_free < masked_need:
        raise SystemExit(f"/tmp has {disk_free / 2**30:.1f} GiB free, masked images need {masked_need / 2**30:.1f} GiB")
    t = time.time()
    n = yolo_data.mask_images(root, masked_seqs, MASKED, frames=20 if smoke else None)
    log("train", step="mask", sequences=masked_seqs, images=n, seconds=round(time.time() - t, 1),
        tmp_free_gib=round(shutil.disk_usage("/tmp").free / 2**30, 1))
batch = 2 if LOCAL else 16
lock = threading.Lock()

# RAM cache of the training images. Ultralytics checks the free RAM once when a job starts, so two jobs that start
# together both see the RAM free and can together fill more than there is. A shared budget prevents that: a job
# caches its images only if its share fits; otherwise it reads them from disk (slower, same result).
FRAME_BYTES = 1280 * 386 * 3  # one KITTI frame (1242 x 375) resized to imgsz 1280, uint8 RGB
# RAM a training job uses besides its image cache (process, 4 train + val data loader workers). Measured on
# Kaggle 2026-10-14 (31.3 GiB total): two jobs caching 8.8 GiB each were killed when training started, two caching
# 6.3 + 5.7 GiB were not, so the overhead lies between 6.9 and 9.7 GiB per job; 10 GiB covers that range.
JOB_OVERHEAD = 10 * 2**30
budget = {"free": mem.available}


def reserve_cache(n_frames):
    """Whether the job caches its n_frames in RAM, and the bytes it holds from the budget until it ends: its
    overhead always, its cache only when both fit in what is left."""
    with lock:
        cache = not smoke and n_frames * FRAME_BYTES + JOB_OVERHEAD <= budget["free"]
        held = JOB_OVERHEAD + (n_frames * FRAME_BYTES if cache else 0)
        budget["free"] -= held
        return cache, held


def release_cache(n):
    with lock:
        budget["free"] += n


def run_job(name, part, seqs, cfg, device):
    cwd = Path(f"/tmp/job_{name}")  # Ultralytics writes into the working directory: one per job
    cwd.mkdir(exist_ok=True)
    (cwd / "yolo26n.pt").unlink(missing_ok=True)
    (cwd / "yolo26n.pt").symlink_to(coco_pt)
    env = f"cd {cwd} && CUDA_VISIBLE_DEVICES={'' if device == 'cpu' else device} PYTHONPATH={SRC}"
    if part is None:
        weights, extra = coco_pt, " --coco"
    else:
        t = time.time()
        with lock:  # yolo_data writes the dataset once per job; keep its file creation serialized
            data = yolo_data.build(root, f"/tmp/yolo_{name}", split=part, splits=("train", "val"),
                                   frames=20 if smoke else None,
                                   masked=MASKED if finetune.CONFIGS[cfg].get("dontcare_masked") else None)
        n_frames = sum(20 if smoke else kitti.FRAMES[q] for p in ("train", "val") for q in part[p])
        cache, held = reserve_cache(n_frames)
        args = f"--data {data} --weights {coco_pt} --project {WORK}/runs --name {name} --config {cfg} --batch {batch}"
        args += f" --cache {'ram' if cache else 'False'}" + (" --epochs 1" if smoke else "")
        args += " --workers 2" if LOCAL else ""
        try:
            sh(f"{env} python -m monodist.finetune {args}", timeout=remaining())
        finally:
            release_cache(held)
        log("train", job=name, step="finetune", device=device, seconds=round(time.time() - t, 1),
            cache_gib=round((held - JOB_OVERHEAD) / 2**30, 1), frames=n_frames)
        weights, extra = WORK / "runs" / name / "weights" / "best.pt", ""
        (weights.parent / "last.pt").unlink(missing_ok=True)  # only best.pt is used; saves output space
    for head, flag in (("nms", " --nms"), ("e2e", "")):
        t = time.time()
        sh(f"{env} python -m monodist.detect --weights {weights} --root {root} --out {WORK}/det/{name}_{head} "
           f"--seqs {' '.join(seqs)}{flag}{extra}" + (" --frames 20" if smoke else ""), timeout=remaining())
        log("train", job=name, step="detect", head=head, device=device, seconds=round(time.time() - t, 1))


todo = queue.Queue()
for j in jobs:
    todo.put(j)
failed = []


def worker(device):
    while True:
        try:
            name, part, seqs, cfg = todo.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(name, part, seqs, cfg, device)
            log("train", job=name, step="done", device=device)
        except Exception:  # the jobs are independent: record the failure and go on with the next one
            failed.append(name)
            log("train", job=name, step="failed", device=device, error=traceback.format_exc()[-2000:])


threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
for th in threads:
    th.start()
for th in threads:
    th.join()
log("train", step="end", failed=failed, seconds=round(time.time() - START, 1))
if failed:
    raise SystemExit(f"failed jobs: {failed}")
