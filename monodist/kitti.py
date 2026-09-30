"""KITTI tracking: labels, calibration, the fixed sequence split and the three cross-validation folds."""
from pathlib import Path

import numpy as np

# Fixed split by sequence (never by frame: neighboring frames are near duplicates).
# Chosen so that Car and Pedestrian are each close to 60/20/20 in instances and in tracks.
SPLIT = {
    "train": ["0001", "0005", "0008", "0009", "0013", "0018", "0019", "0020"],
    "val": ["0003", "0004", "0006", "0011", "0016"],
    "test": ["0000", "0002", "0007", "0010", "0012", "0014", "0015", "0017"],
}
SEQUENCES = sorted(s for seqs in SPLIT.values() for s in seqs)
# Three-fold cross-validation by sequence: every sequence is test exactly once. Folds balance cars (instances and
# tracks) and pedestrian tracks; pedestrian instances cannot be balanced, sequence 0019 alone holds 53% of them.
# Within each fold the non-test sequences split about 70/30 into train and val the same way.
FOLDS = [
    {"test": ["0002", "0006", "0007", "0009", "0012", "0013", "0014", "0017", "0018"],
     "val": ["0001", "0008", "0015", "0016"],
     "train": ["0000", "0003", "0004", "0005", "0010", "0011", "0019", "0020"]},
    {"test": ["0003", "0005", "0008", "0019", "0020"],
     "val": ["0000", "0004", "0006", "0007", "0010", "0015", "0017"],
     "train": ["0001", "0002", "0009", "0011", "0012", "0013", "0014", "0016", "0018"]},
    {"test": ["0000", "0001", "0004", "0010", "0011", "0015", "0016"],
     "val": ["0002", "0003", "0005", "0006", "0008", "0013", "0018"],
     "train": ["0007", "0009", "0012", "0014", "0017", "0019", "0020"]},
]
CLASSES = ("Car", "Pedestrian")
EXTRA_CLASSES = ("Van", "Cyclist")  # optionally labeled when fine-tuning; never evaluated
# A detection on a neighboring class counts neither as a hit nor as a false alarm (as in the KITTI devkit).
NEIGHBORS = {"Van": "Car", "Person": "Pedestrian"}


def training_dir(root):
    """Accept either the dataset root or its training/ directory."""
    root = Path(root)
    for cand in (root, root / "training", root / "kitti_tracking" / "training"):
        if (cand / "label_02").is_dir():
            return cand
    raise FileNotFoundError(f"no label_02/ under {root}")


def load_labels(root, seq):
    """One sequence of label_02 as a dict of arrays, one row per labeled object per frame."""
    rows = [line.split() for line in open(training_dir(root) / "label_02" / f"{seq}.txt")]
    num = np.array([[float(v) for i, v in enumerate(r) if i != 2] for r in rows]).reshape(-1, 16)
    return {
        "frame": num[:, 0].astype(int),
        "track": num[:, 1].astype(int),
        "type": np.array([r[2] for r in rows]),
        "trunc": num[:, 2].astype(int),
        "occ": num[:, 3].astype(int),
        "box": num[:, 5:9],          # x1, y1, x2, y2 in pixels (camera 2 image)
        "dim": num[:, 9:12],         # height, width, length in meters
        "loc": num[:, 12:15],        # bottom center of the 3D box, rectified camera-0 frame
        "ry": num[:, 15],
    }


def load_calib(root, seq):
    """Projection matrix P2 (3x4) of the left color camera."""
    for line in open(training_dir(root) / "calib" / f"{seq}.txt"):
        if line.startswith("P2:"):
            return np.array(line.split()[1:], float).reshape(3, 4)
    raise ValueError(f"no P2 in calib {seq}")


def intrinsics(P2):
    """f, cx, cy and the offset t of camera 2 from the rectified camera-0 frame (P2 = K [I | t])."""
    K = P2[:, :3]
    t = np.linalg.solve(K, P2[:, 3])
    return {"f": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "t": t}


def to_camera2(loc, P2):
    """Move label locations from the camera-0 frame into camera 2, whose image the boxes live in."""
    return loc + intrinsics(P2)["t"]


def project(points, P2):
    """Project (N, 3) camera-0 points to (N, 2) pixels of camera 2."""
    p = np.hstack([points, np.ones((len(points), 1))]) @ P2.T
    return p[:, :2] / p[:, 2:3]


def box_corners(dim, loc, ry):
    """The 8 corners (8, 3) of one labeled 3D box."""
    h, w, l = dim
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    c, s = np.cos(ry), np.sin(ry)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (R @ np.vstack([x, y, z])).T + loc


def image_path(root, seq, frame):
    return training_dir(root) / "image_02" / seq / f"{frame:06d}.png"


# Images per training sequence (8,008 in total), from the dataset listing.
FRAMES = {"0000": 154, "0001": 447, "0002": 233, "0003": 144, "0004": 314, "0005": 297, "0006": 270,
          "0007": 800, "0008": 390, "0009": 803, "0010": 294, "0011": 373, "0012": 78, "0013": 340,
          "0014": 106, "0015": 376, "0016": 209, "0017": 145, "0018": 339, "0019": 1059, "0020": 837}
