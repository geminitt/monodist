"""Per-frame latency of the whole pipeline, stage by stage, one frame at a time (batch 1), on the GPU.

    python -m monodist.latency --weights best.pt --imgsz 1280 --root <kitti> --out results/latency/ft_1280.json

Stages: read the PNG file, decode it, then Ultralytics' own synchronised timers for preprocess (letterbox,
upload), inference and postprocess, then the distance step on CPU. "overhead" is what model.predict costs
beyond its three timed stages. Frames are the unit of the statistics; rounds repeat the same frames.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from . import geometry, kitti
from .detect import parse_imgsz
from .mlp import DistanceMLP

STAGES = ("read", "decode", "preprocess", "inference", "postprocess", "overhead", "distance", "total")


def distance_step(model, boxes, cls, cam):
    """What the pipeline does after detection: features, the MLP and both formulas for every kept box."""
    f, cx, cy, w, h = cam
    x = geometry.features(boxes, cls, f, cx, cy, w, h)
    with torch.no_grad():
        z = torch.exp(model(torch.tensor(x, dtype=torch.float32))).numpy()
    geometry.size_distance(boxes, cls, f, [1.5, 1.75])
    geometry.ground_distance(boxes, f, cy, 1.6, h)
    return z


def bootstrap_median(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return [float(np.median(v)), *np.percentile(meds, [2.5, 97.5]).tolist()]


def run(weights, root, seq, frames, rounds, warmup, imgsz, half, pipelined=False):
    from ultralytics import YOLO
    model = YOLO(weights, task="detect")
    mlp = DistanceMLP(len(geometry.FEATURES)).eval()
    torch.set_num_threads(1)
    k = kitti.intrinsics(kitti.load_calib(root, seq))
    paths = [kitti.image_path(root, seq, i) for i in range(frames)]
    coco = len(model.names) == 80
    kw = dict(imgsz=imgsz, quantize=16 if half else None, conf=0.25, max_det=300, nms=False, verbose=False,
              classes=[0, 2] if coco else None)

    def load(p):
        t0 = time.perf_counter()
        data = np.frombuffer(open(p, "rb").read(), np.uint8)
        t1 = time.perf_counter()
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img, t1 - t0, time.perf_counter() - t1

    def process(img, t_start, t_read, t_decode):
        t2 = time.perf_counter()
        r = model.predict(img, **kw)[0]
        t3 = time.perf_counter()
        boxes, cls = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        if coco:
            cls = (cls == 0).astype(int)  # COCO person -> 1 (Pedestrian), car -> 0 (Car)
        distance_step(mlp, boxes, cls, (k["f"], k["cx"], k["cy"], img.shape[1], img.shape[0]))
        t4 = time.perf_counter()
        sp = r.speed
        return {"read": t_read, "decode": t_decode, "preprocess": sp["preprocess"] / 1e3,
                "inference": sp["inference"] / 1e3, "postprocess": sp["postprocess"] / 1e3,
                "overhead": (t3 - t2) - sum(sp.values()) / 1e3, "distance": t4 - t3, "total": t4 - t_start}

    for p in paths[:warmup]:
        process(*load(p)[:1], time.perf_counter(), 0, 0)
    records = []
    wall = []
    for _ in range(rounds):
        t_round = time.perf_counter()
        if pipelined:  # decode the next frame on a worker thread while the GPU works on this one
            with ThreadPoolExecutor(1) as pool:
                nxt = pool.submit(load, paths[0])
                for i in range(len(paths)):
                    t_start = time.perf_counter()
                    img, tr, td = nxt.result()
                    if i + 1 < len(paths):
                        nxt = pool.submit(load, paths[i + 1])
                    records.append(process(img, t_start, tr, td))
        else:
            for p in paths:
                t_start = time.perf_counter()
                img, tr, td = load(p)
                records.append(process(img, t_start, tr, td))
        wall.append(time.perf_counter() - t_round)
    out = {"weights": str(weights), "imgsz": imgsz, "half": half, "pipelined": pipelined, "seq": seq,
           "frames": frames, "rounds": rounds, "fps_per_round": [frames / w for w in wall]}
    for s in STAGES:
        v = np.array([r[s] for r in records]) * 1e3
        per_round = v.reshape(rounds, frames)
        out[s] = {"median_ms": bootstrap_median(v), "p90_ms": float(np.percentile(v, 90)),
                  "round_medians_ms": np.median(per_round, axis=1).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="0007")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--imgsz", default="1280")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--pipelined", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print("exists, skipping:", out)
        return
    res = run(args.weights, args.root, args.seq, args.frames, args.rounds, args.warmup, parse_imgsz(args.imgsz),
              args.half, args.pipelined)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(out.name, {s: round(res[s]["median_ms"][0], 2) for s in STAGES}, "fps", [round(v, 1) for v in res["fps_per_round"]])


if __name__ == "__main__":
    main()
