"""Write the KITTI split in the YOLO dataset layout (image symlinks + normalised box labels) for fine-tuning.

Only Car and Pedestrian are labelled; the classes are 0 = Car, 1 = Pedestrian.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import kitti


def yolo_lines(boxes, classes, w, h):
    """Box rows 'cls cx cy bw bh', normalised to [0, 1], boxes clipped to the image."""
    b = np.asarray(boxes, float).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    lines = []
    for (x1, y1, x2, y2), c in zip(b, classes):
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def build(root, out, splits=("train", "val", "test"), frames=None, only=None):
    out = Path(out)
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for seq in [s for s in kitti.SPLIT[split] if only is None or s in only]:
            lab = kitti.load_labels(root, seq)
            w, h = Image.open(kitti.image_path(root, seq, 0)).size
            for fr in range(frames or kitti.FRAMES[seq]):
                name = f"{seq}_{fr:06d}"
                link = out / "images" / split / f"{name}.png"
                if not link.exists():
                    os.symlink(kitti.image_path(root, seq, fr).resolve(), link)
                m = (lab["frame"] == fr) & np.isin(lab["type"], kitti.CLASSES)
                cls = [kitti.CLASSES.index(t) for t in lab["type"][m]]
                (out / "labels" / split / f"{name}.txt").write_text("\n".join(yolo_lines(lab["box"][m], cls, w, h)))
    yaml = out / "kitti.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    f"names:\n  0: Car\n  1: Pedestrian\n")
    return yaml


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(build(args.root, args.out))
