"""Write a KITTI split in the YOLO dataset layout (images + normalized box labels) for fine-tuning.

Class ids follow kitti.CLASSES: 0 = Car, 1 = Van, 2 = LargeVehicle (Truck, Tram), 3 = Pedestrian (Pedestrian,
Person), 4 = Cyclist. Misc and DontCare stay unlabeled. Images are symlinked, not copied.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import kitti


def yolo_lines(boxes, classes, w, h):
    """Box rows 'cls cx cy bw bh', normalized to [0, 1], boxes clipped to the image."""
    b = np.asarray(boxes, float).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    lines = []
    for (x1, y1, x2, y2), c in zip(b, classes):
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def build(root, out, split=None, splits=("train", "val", "test"), frames=None, only=None):
    split = split or kitti.SPLIT
    out = Path(out)
    for part in splits:
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        for seq in [s for s in split[part] if only is None or s in only]:
            lab = kitti.load_labels(root, seq)
            groups = kitti.classify(lab["type"])
            w, h = Image.open(kitti.image_path(root, seq, 0)).size
            for fr in range(frames or kitti.FRAMES[seq]):
                name = f"{seq}_{fr:06d}"
                src = kitti.image_path(root, seq, fr)
                dst = out / "images" / part / f"{name}.png"
                in_frame = lab["frame"] == fr
                if not dst.exists():
                    os.symlink(Path(src).resolve(), dst)
                m = in_frame & (groups >= 0)
                (out / "labels" / part / f"{name}.txt").write_text("\n".join(yolo_lines(lab["box"][m], groups[m], w, h)))
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(kitti.CLASSES))
    yaml = out / "kitti.yaml"
    yaml.write_text(f"path: {out.resolve()}\n" + "".join(f"{p}: images/{p}\n" for p in splits) + f"names:\n{names}")
    return yaml


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(build(args.root, args.out))
