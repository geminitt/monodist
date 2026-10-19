"""Per-frame latency of the whole pipeline, stage by stage, one frame at a time (batch 1), on the GPU.

    python -m monodist.latency --weights best.pt --imgsz 1280 --root <kitti> --out results/latency/ft_1280.json

Stages: read the PNG file, decode it, then Ultralytics' own synchronized timers for preprocess (letterbox,
upload), inference and postprocess, then the distance step on CPU. "overhead" is what model.predict costs
beyond its three timed stages. Frames are the unit of the statistics; rounds repeat the same frames.
With --pipelined the next frame is read and decoded on a worker thread, so "total" becomes the time per frame
in steady state (the inverse of throughput), not the latency of one frame.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from . import detect, kitti, mlp
from .detect import parse_imgsz, weights_identity
from .pipeline import Pipeline

STAGES = ("read", "decode", "preprocess", "inference", "postprocess", "overhead", "distance", "total")


def data_identity(root, seq, frames):
    data = kitti.training_dir(root)
    files = [data / "calib" / f"{seq}.txt"] + [kitti.image_path(root, seq, i) for i in range(frames)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {str(p): weights_identity(p)["sha256"] for p in files}


def hardware_identity():
    return {"cpu": platform.processor(), "cpu_affinity": len(os.sched_getaffinity(0)),
            "torch_cuda_build": torch.version.cuda,
            "gpus": [{"name": torch.cuda.get_device_name(i),
                      "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", "unknown"))}
                     for i in range(torch.cuda.device_count())]}


def bootstrap_median(v, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return [float(np.median(v)), *np.percentile(meds, [2.5, 97.5]).tolist()]


def run(weights, root, seq, frames, rounds, warmup, imgsz, half, pipelined=False, nms=False, compile_mode=None,
        distance=None, temporal_file=None):
    if not distance:
        raise ValueError("a saved trained distance ensemble is required")
    if compile_mode:
        raise ValueError("compiled inference has not been verified with the rebuilt pipeline")
    _, meta = mlp.load(distance)
    settings = json.load(open(temporal_file)) if temporal_file else meta.get("temporal")
    if not settings:
        raise ValueError("validation-selected temporal settings are required")
    signature = meta["detector"]
    if signature is None or signature["weights"]["sha256"] != weights_identity(weights)["sha256"]:
        raise ValueError("detector weights differ from the saved distance model's weights")
    if (str(imgsz), half, nms) != (signature["imgsz"], signature["half"], signature["nms"]):
        raise ValueError("timing configuration differs from the distance model's evaluated detector configuration")
    if frames < 1 or frames > kitti.FRAMES[seq] or rounds < 1 or warmup < 1:
        raise ValueError("invalid frame, round or warmup count")
    pipe = Pipeline(weights, distance, imgsz, nms, signature["coco"], half, temporal_settings=settings)
    torch.set_num_threads(1)
    k = kitti.intrinsics(kitti.load_calib(root, seq))
    paths = [kitti.image_path(root, seq, i) for i in range(frames)]

    def load(p):
        t0 = time.perf_counter()
        data = np.frombuffer(open(p, "rb").read(), np.uint8)
        t1 = time.perf_counter()
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img, t1 - t0, time.perf_counter() - t1

    def process(img, t_start, t_read, t_decode):
        t2 = time.perf_counter()
        r = detect.predict_images(pipe.detector, [img], imgsz, float(pipe.conf.min()), pipe.coco, nms, half)[0]
        t3 = time.perf_counter()
        pipe.finish(r, (k["f"], k["cx"], k["cy"]))
        t4 = time.perf_counter()
        sp = r["speed"]
        return {"read": t_read, "decode": t_decode, "preprocess": sp["preprocess"] / 1e3,
                "inference": sp["inference"] / 1e3, "postprocess": sp["postprocess"] / 1e3,
                "overhead": (t3 - t2) - sum(sp.values()) / 1e3, "distance": t4 - t3, "total": t4 - t_start}

    for p in paths[:warmup]:
        process(*load(p)[:1], time.perf_counter(), 0, 0)
    if isinstance(imgsz, list):  # a fixed-shape engine silently runs at its own shape: make sure it is the asked one
        assert list(pipe.detector.predictor.imgsz) == imgsz, f"engine runs at {pipe.detector.predictor.imgsz}, asked for {imgsz}"
    records = []
    wall = []
    for _ in range(rounds):
        pipe.reset()
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
    out = {"weights": str(weights), "imgsz": imgsz, "half": half, "pipelined": pipelined, "nms": nms,
           "compile": compile_mode, "seq": seq,
           "distance_model": weights_identity(distance), "temporal": settings, "thresholds": meta["conf"],
           "includes_tracking_and_ttc": True,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "frames": frames, "rounds": rounds, "fps_per_round": [frames / w for w in wall]}
    for s in STAGES:
        v = np.array([r[s] for r in records]) * 1e3
        per_round = v.reshape(rounds, frames)
        out[s] = {"median_ms": bootstrap_median(v), "p90_ms": float(np.percentile(v, 90)),
                  "round_medians_ms": np.median(per_round, axis=1).tolist()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--distance", required=True, help="saved trained ensemble paired with these detector weights")
    ap.add_argument("--temporal", help="JSON with validation-selected meas_rel and accel (unless in the model)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="0019")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--imgsz", default="1280")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--pipelined", action="store_true")
    ap.add_argument("--nms", action="store_true")
    ap.add_argument("--compile", default=None, help='torch.compile mode, e.g. "reduce-overhead" (CUDA graphs)')
    args = ap.parse_args(argv)
    out = Path(args.out)
    signature = {"weights": weights_identity(args.weights), "distance": weights_identity(args.distance),
                 "temporal": weights_identity(args.temporal) if args.temporal else None,
                 "source": {name: weights_identity(Path(__file__).with_name(name))["sha256"]
                            for name in ("latency.py", "pipeline.py", "detect.py", "geometry.py", "kitti.py", "mlp.py", "temporal.py")},
                 "inputs": data_identity(args.root, args.seq, args.frames),
                 "versions": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "ultralytics")},
                 "hardware": hardware_identity(),
                 **{k: v for k, v in vars(args).items() if k not in ("weights", "distance", "temporal", "root", "out")}}
    if out.exists():  # skip only a timing of the same weights and settings
        old = json.loads(out.read_text()).get("signature")
        if old != signature:
            raise SystemExit(f"{out} holds a timing of another configuration ({old}); use another --out")
        print("exists, skipping:", out)
        return
    res = run(args.weights, args.root, args.seq, args.frames, args.rounds, args.warmup, parse_imgsz(args.imgsz),
              args.half, args.pipelined, args.nms, args.compile, args.distance, args.temporal)
    res["signature"] = signature
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(out.name, {s: round(res[s]["median_ms"][0], 2) for s in STAGES}, "fps", [round(v, 1) for v in res["fps_per_round"]])


if __name__ == "__main__":
    main()
