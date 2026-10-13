"""KITTI tracking: labels, calibration, classes, the held-out test sequences and the cross-validation folds."""
from pathlib import Path

import numpy as np

# Splits are by sequence, never by frame: neighboring frames are near duplicates.
# TEST is held out: it is opened once, at the end, with the final model; no choice may read it.
# It holds 17-25% of each reporting group's instances and tracks (19% of the Cyclist tracks: 7 of 37).
TEST = ["0003", "0007", "0011", "0016", "0017"]
# Three-fold cross-validation by sequence on the other 16: every one of them is test exactly once, and every
# choice (configuration, epoch, threshold, filter setting) is made on these folds. Within a fold the non-test
# sequences split about 70/30 into train and val. Person instances cannot be balanced: sequence 0019 alone
# holds 72% of those outside TEST. Found by a deterministic search over the group counts.
FOLDS = [
    {"test": ["0001", "0004", "0006", "0010", "0014", "0015"],
     "val": ["0009", "0013", "0018"],
     "train": ["0000", "0002", "0005", "0008", "0012", "0019", "0020"]},
    {"test": ["0005", "0019", "0020"],
     "val": ["0000", "0002", "0004", "0010", "0012", "0014"],
     "train": ["0001", "0006", "0008", "0009", "0013", "0015", "0018"]},
    {"test": ["0000", "0002", "0008", "0009", "0012", "0013", "0018"],
     "val": ["0001", "0010", "0014", "0015"],
     "train": ["0004", "0005", "0006", "0019", "0020"]},
]
# The final model trains on the 16 non-test sequences (val chooses its epoch and thresholds) and is scored on TEST.
SPLIT = {
    "train": ["0000", "0001", "0004", "0006", "0008", "0010", "0012", "0015", "0018", "0019", "0020"],
    "val": ["0002", "0005", "0009", "0013", "0014"],
    "test": TEST,
}
SEQUENCES = sorted(s for seqs in SPLIT.values() for s in seqs)
DEVELOPMENT = sorted(set(SEQUENCES) - set(TEST))
# Detected classes and the KITTI types each one gathers. Distance from a box depends on the object's real size,
# so the classes keep sizes apart (one Vehicle class put trucks and cars together and cost 3 points of AbsRel on
# label boxes); Truck and Tram share a class because Tram appears in 3 sequences only.
CLASSES = ("Car", "Van", "LargeVehicle", "Pedestrian", "Cyclist")
TYPES = {"Car": 0, "Van": 1, "Truck": 2, "Tram": 2, "Pedestrian": 3, "Person": 3, "Cyclist": 4}
# Reporting groups: class index -> group index.
GROUPS = ("Vehicle", "Person", "Cyclist")
GROUP_OF = np.array([0, 0, 0, 1, 2])
# Misc (an ambiguous label) is an ignore region: a detection on it counts neither as a hit nor as a false alarm,
# and missing it is not a miss. DontCare regions are handled the same way, by area coverage (as in the devkit).
IGNORE = ("Misc",)


def classify(types):
    """Class index of each KITTI type, -1 for types outside the classes."""
    return np.array([TYPES.get(t, -1) for t in types], int)


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


def center_range(dim, loc_cam2):
    """Euclidean distance from camera 2 to the center of each 3D box (loc is the bottom center; y points down)."""
    c = np.asarray(loc_cam2, float).copy()
    c[:, 1] -= np.asarray(dim)[:, 0] / 2
    return np.linalg.norm(c, axis=1)


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
