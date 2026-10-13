"""Run a YOLO detector over KITTI sequences and save every detection above a low score.

    python -m monodist.detect --weights yolo26n.pt --coco --root <kitti> --out results/det/coco_1280

One .npz per sequence, so an interrupted run resumes where it stopped; merge() joins them. The folder's
run.json records the weights (SHA-256) and settings: a rerun resumes only with the same ones.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import kitti


def parse_imgsz(s):
    v = [int(x) for x in str(s).split(",")]
    return v[0] if len(v) == 1 else v

# COCO weights: car -> Car; bus, train and truck -> LargeVehicle; person -> Pedestrian. COCO has no van and no
# cyclist class (a rider is a person), so the COCO baseline never predicts Van or Cyclist.
COCO_TO_KITTI = {2: 0, 5: 2, 6: 2, 7: 2, 0: 3}


def predict_images(model, images, imgsz, conf, coco=False, nms=False, half=False):
    """Detections of a batch of images (paths or arrays): one dict per image with box, score, cls and image_hw."""
    # nms=False selects YOLO26's one-to-one (NMS-free) head, nms=None its one-to-many head followed by NMS;
    # quantize=16 runs the model in fp16
    results = model.predict(images, imgsz=imgsz, quantize=16 if half else None, conf=conf,
                            classes=list(COCO_TO_KITTI) if coco else None, max_det=300,
                            nms=None if nms else False, verbose=False)
    out = []
    for r in results:
        b = r.boxes
        cls = b.cls.cpu().numpy().astype(int)
        if coco:
            cls = np.array([COCO_TO_KITTI[c] for c in cls], int)
        out.append({"box": b.xyxy.cpu().numpy().reshape(-1, 4), "score": b.conf.cpu().numpy(), "cls": cls,
                    "image_hw": np.array(r.orig_shape)})
    return out


def detect_sequence(model, root, seq, imgsz, half, conf, coco, batch, frames=None, nms=False):
    paths = [str(kitti.image_path(root, seq, f)) for f in range(frames or kitti.FRAMES[seq])]
    rows = {"frame": [], "box": [], "score": [], "cls": []}
    shape = None
    for i in range(0, len(paths), batch):
        for k, r in enumerate(predict_images(model, paths[i:i + batch], imgsz, conf, coco, nms, half)):
            shape = r["image_hw"]
            rows["frame"].append(np.full(len(r["cls"]), i + k))
            for key in ("box", "score", "cls"):
                rows[key].append(r[key])
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["image_hw"] = np.array(shape)
    return out


def merge(out_dir, seqs=None):
    """All per-sequence files of one run as one dict of arrays, plus image sizes per sequence."""
    out_dir = Path(out_dir)
    # <seq>.tmp.npz is a file still being written (or left by a stopped run): never part of the results
    seqs = seqs or sorted(p.stem for p in out_dir.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    parts, sizes = [], {}
    for s in seqs:
        d = dict(np.load(out_dir / f"{s}.npz"))
        sizes[s] = tuple(int(v) for v in d.pop("image_hw"))
        d["seq"] = np.full(len(d["frame"]), s)
        parts.append(d)
    det = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    det["box"] = det["box"].reshape(-1, 4)
    return det, sizes


def weights_identity(weights):
    """Name and SHA-256 of a weights or engine file (sha256 None if the file is not there)."""
    p = Path(weights)
    sha = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return {"name": p.name, "sha256": sha}


def check_signature(out_dir, signature):
    """Record the run's configuration in out_dir/run.json, or stop if the folder holds another configuration.

    Outputs are skipped when they exist; without this check a rerun with other weights or settings would silently
    keep the old files. A folder with detections but no run.json predates the check and is not resumed either.
    """
    f = Path(out_dir) / "run.json"
    if f.exists():
        old = json.loads(f.read_text())
        if old != signature:
            raise SystemExit(f"{out_dir} holds a run with another configuration:\n  {old}\nnot\n  {signature}\n"
                             "use another --out")
        return
    if any(Path(out_dir).glob("*.npz")):
        raise SystemExit(f"{out_dir} holds detections but no run.json: their configuration is unknown; use another --out")
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(signature, indent=1))
    tmp.rename(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coco", action="store_true", help="COCO-trained weights: keep car and person only")
    ap.add_argument("--imgsz", default="1280", help="one size, or H,W for a fixed-shape TensorRT engine")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--nms", action="store_true", help="one-to-many head + NMS instead of the NMS-free head")
    ap.add_argument("--seqs", nargs="*", default=kitti.SEQUENCES)
    ap.add_argument("--frames", type=int, default=None, help="only the first N frames (smoke tests)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # after loading: Ultralytics downloads a missing official checkpoint (yolo26n.pt) to this path
    check_signature(out, {"weights": weights_identity(args.weights), "coco": args.coco, "imgsz": args.imgsz,
                          "half": args.half, "conf": args.conf, "batch": args.batch, "nms": args.nms,
                          "frames": args.frames})
    for seq in args.seqs:
        f = out / f"{seq}.npz"
        if f.exists():
            continue
        t = time.time()
        d = detect_sequence(model, args.root, seq, parse_imgsz(args.imgsz), args.half, args.conf, args.coco, args.batch, args.frames,
                             args.nms)
        np.savez(out / f"{seq}.tmp.npz", **d)
        (out / f"{seq}.tmp.npz").rename(f)
        print(f"{seq}: {len(d['score'])} detections in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
