"""Is YOLO26n inference in PyTorch limited by kernel launches? Count the CUDA kernels of one forward pass, sum
their GPU time, and compare with the wall-clock time of the forward pass, then replay it as a CUDA graph.

    python -m monodist.profile_gpu --weights best.pt --out results/latency/profile.json

If the GPU is busy for only a small part of the wall time, the rest is the CPU launching kernels one by one.
A CUDA graph records the whole sequence of kernels once and replays it with a single launch, so it removes
exactly that cost: if the hypothesis holds, the graph replay is far faster than eager PyTorch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


def wall_ms(fn, n=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1e3)
    return float(np.median(times))


def kernels(fn, n=20):
    """CUDA kernels launched and GPU time spent per call, from the PyTorch profiler."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    busy = sum(getattr(e, "device_time", getattr(e, "cuda_time", 0.0)) for e in ev) / 1e3  # us -> ms
    return len(ev) / n, busy / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", default="416,1280", help="H,W of the letterboxed input")
    ap.add_argument("--nms", action="store_true", help="time the one-to-many head (as used with NMS)")
    args = ap.parse_args()

    from ultralytics import YOLO
    h, w = (int(v) for v in args.imgsz.split(","))
    net = YOLO(args.weights).model.fuse().eval().cuda()
    net.model[-1].end2end = not args.nms
    res = {"weights": args.weights, "imgsz": [h, w], "nms_head": args.nms}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        model = net.half() if dtype == torch.float16 else net.float()
        x = torch.rand(1, 3, h, w, device="cuda", dtype=dtype)
        with torch.no_grad():
            eager = lambda: model(x)
            n_kernels, busy = kernels(eager)
            row = {"kernels_per_forward": n_kernels, "gpu_busy_ms": busy, "eager_wall_ms": wall_ms(eager)}
            try:
                static = x.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):  # warm up on a side stream before capture, as the CUDA docs ask
                    for _ in range(3):
                        model(static)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    model(static)
                row["graph_wall_ms"] = wall_ms(graph.replay)
            except Exception as e:  # noqa: BLE001
                row["graph_error"] = repr(e)[:500]
        row["gpu_busy_share"] = row["gpu_busy_ms"] / row["eager_wall_ms"]
        res[name] = row
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
