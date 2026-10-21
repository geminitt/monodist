"""Write a KITTI split in the YOLO dataset layout (images + normalized box labels) for fine-tuning.

Class ids follow kitti.CLASSES: 0 = Car, 1 = Van, 2 = LargeVehicle (Truck, Tram), 3 = Pedestrian (Pedestrian,
Person), 4 = Cyclist. Misc and DontCare stay unlabeled. Images are symlinked, not copied.

With masked=<dir>, each image with a DontCare region is replaced by a copy in which the DontCare pixels are
filled with gray, except pixels inside any other labeled box. KITTI marks objects it did not label (mostly far
or crowded ones) as DontCare; unmasked, training treats them as background, although evaluation ignores them.
Detection and evaluation always use the original images.
"""
import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
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


FILL = 114  # the gray Ultralytics pads letterboxed images with


def dontcare_mask(boxes, types, w, h):
    """Pixels to fill in one frame: inside a DontCare box and outside every other labeled box (Misc included)."""
    mask = np.zeros((h, w), bool)

    def pixels(b):  # the pixels a box touches, clipped to the image
        x1, y1 = max(0, math.floor(b[0])), max(0, math.floor(b[1]))
        return slice(y1, max(y1, min(h, math.ceil(b[3])))), slice(x1, max(x1, min(w, math.ceil(b[2]))))

    for b in boxes[types == "DontCare"]:
        mask[pixels(b)] = True
    for b in boxes[types != "DontCare"]:
        mask[pixels(b)] = False
    return mask


def masked_image(root, seq, fr, lab, masked):
    """Path of the frame's image to train on: the masked copy in masked/ (written once), or the original."""
    src = kitti.image_path(root, seq, fr)
    m = lab["frame"] == fr
    if not (lab["type"][m] == "DontCare").any():
        return Path(src).resolve()
    dst = Path(masked) / f"{seq}_{fr:06d}.png"
    if not dst.exists():
        img = np.array(Image.open(src).convert("RGB"))
        img[dontcare_mask(lab["box"][m], lab["type"][m], img.shape[1], img.shape[0])] = FILL
        tmp = dst.with_name(dst.stem + ".part.png")  # a cut write never leaves a file under the final name
        Image.fromarray(img).save(tmp, compress_level=1)
        tmp.rename(dst)
    return dst.resolve()


def mask_images(root, seqs, masked, frames=None, workers=None):
    """Write the masked copies of every frame of seqs in parallel; returns how many images now exist there."""
    Path(masked).mkdir(parents=True, exist_ok=True)
    work = [(seq, fr, lab) for seq in seqs for lab in [kitti.load_labels(root, seq)]
            for fr in range(frames or kitti.FRAMES[seq])]
    with ThreadPoolExecutor(workers or os.cpu_count()) as pool:
        list(pool.map(lambda job: masked_image(root, *job, masked), work))
    return sum(1 for p in Path(masked).glob("*.png") if not p.name.endswith(".part.png"))


def build(root, out, split=None, splits=("train", "val", "test"), frames=None, only=None, masked=None):
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
                target = masked_image(root, seq, fr, lab, masked) if masked else Path(src).resolve()
                if dst.is_symlink() and dst.resolve() != target:  # left by a build with the other masking
                    dst.unlink()
                if not dst.exists():
                    os.symlink(target, dst)
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
