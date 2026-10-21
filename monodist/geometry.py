"""Distance from one camera: the two geometric methods and the features of the learned one.

Distances are depths Z along the optical axis of camera 2, in meters; depth_to_range turns one into the
Euclidean distance.
Boxes are (N, 4) arrays of x1, y1, x2, y2 in pixels; classes are indices into kitti.CLASSES.
"""
import numpy as np

from .kitti import CLASSES

BORDER = 2.0  # a box edge this close to the image border (pixels) means the object is cut off


def iou(a, b):
    """Pairwise IoU between (N, 4) and (M, 4) boxes -> (N, M)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def iou_pairs(a, b):
    """Row-wise IoU of two (N, 4) box arrays."""
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def size_distance(box, cls, f, class_height):
    """Known-size method: Z = f * H / h, with H the mean real height of the class."""
    h = np.maximum(box[:, 3] - box[:, 1], 1e-6)
    return f * np.asarray(class_height)[cls] / h


def ground_distance(box, f, cy, cam_height, image_h):
    """Ground-plane method: the bottom edge touches a flat road cam_height below a level camera.

    v2 = cy + f * cam_height / Z  =>  Z = f * cam_height / (v2 - cy).
    NaN where the method does not apply: box cut by the bottom border, or bottom edge at or above the horizon.
    """
    v2 = box[:, 3]
    below = v2 - cy
    ok = (below > 1.0) & (v2 < image_h - BORDER)
    return np.where(ok, f * cam_height / np.where(ok, below, 1.0), np.nan)


def ground_or_size(box, cls, f, cy, cam_height, image_h, class_height):
    """Ground-plane distance, falling back to the known-size one where the ground plane does not apply."""
    zg = ground_distance(box, f, cy, cam_height, image_h)
    fallback = np.isnan(zg)
    return np.where(fallback, size_distance(box, cls, f, class_height), zg), fallback


# One-hot class columns: a single class number would tell the MLP that Cyclist is "more" than Pedestrian.
_BASE = ("h/f", "w/f", "(v2-cy)/f", "(v1-cy)/f", "(u-cx)/f", *(c.lower() for c in CLASSES))
# "cut": one flag for a box at any image border (the ensembles timed and TEST-scored on 2026-10-19).
# "edge": one flag per border. A box cut at the bottom has lost its ground contact, a box cut at a side has not;
# development out-of-fold, edge flags lowered the near-car (0-10 m) mean absolute error from 1.53 to 1.21 m.
FEATURE_SETS = {"cut": _BASE + ("cut",), "edge": _BASE + ("cut_left", "cut_right", "cut_top", "cut_bottom")}
FEATURES = FEATURE_SETS["edge"]  # what new ensembles are trained on
N_CLASSES = len(CLASSES)


def feature_names(meta):
    """The feature set a saved ensemble was trained on; refuses a set this code cannot compute."""
    names = tuple(meta["features"])
    if names not in FEATURE_SETS.values():
        raise ValueError(f"unknown distance features {list(names)}")
    return names


def features(box, cls, f, cx, cy, image_w, image_h, names=FEATURES):
    """Inputs of the learned distance model, every pixel quantity divided by f (angles, camera independent)."""
    if tuple(names) not in FEATURE_SETS.values():
        raise ValueError(f"unknown distance features {list(names)}")
    x1, y1, x2, y2 = box.T
    edges = [x1 < BORDER, x2 > image_w - BORDER, y1 < BORDER, y2 > image_h - BORDER]  # left, right, top, bottom
    flags = edges if names[-1] != "cut" else [edges[0] | edges[1] | edges[2] | edges[3]]
    return np.stack([
        (y2 - y1) / f,
        (x2 - x1) / f,
        (y2 - cy) / f,
        (y1 - cy) / f,
        ((x1 + x2) / 2 - cx) / f,
        *np.eye(N_CLASSES)[np.asarray(cls, int)].T,
        *(flag.astype(float) for flag in flags),
    ], axis=1)


def depth_to_range(box, z, f, cx, cy):
    """Euclidean distance along the ray through the box center, from the depth Z of that point (fy = fx assumed)."""
    u, v = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    return z * np.sqrt(1 + ((u - cx) / f) ** 2 + ((v - cy) / f) ** 2)
