import numpy as np

from monodist import kitti
from monodist.geometry import iou


def test_split_is_a_partition_of_the_21_sequences():
    seqs = [s for v in kitti.SPLIT.values() for s in v]
    assert len(seqs) == len(set(seqs)) == 21
    assert sorted(seqs) == sorted(kitti.FRAMES)
    assert sum(kitti.FRAMES.values()) == 8008


def test_labels_parse(root):
    for seq in kitti.SEQUENCES:
        lab = kitti.load_labels(root, seq)
        n = len(lab["frame"])
        assert all(len(v) == n for v in lab.values())
        assert lab["frame"].max() < kitti.FRAMES[seq]
        obj = np.isin(lab["type"], kitti.CLASSES)
        assert (lab["dim"][obj] > 0).all()
        assert (lab["box"][obj, 2] > lab["box"][obj, 0]).all() and (lab["box"][obj, 3] > lab["box"][obj, 1]).all()


def test_intrinsics_rebuild_P2(root):
    P2 = kitti.load_calib(root, "0000")
    k = kitti.intrinsics(P2)
    K = P2[:, :3]
    assert np.allclose(K @ k["t"], P2[:, 3])
    assert abs(k["f"] - 721.5377) < 1e-3 and abs(k["t"][0] - 0.0597) < 1e-3  # camera 2 sits ~6 cm left of camera 0


def test_projected_3d_boxes_match_2d_boxes(root):
    """The 3D labels, projected with P2, should land on the 2D labels for fully visible objects."""
    lab = kitti.load_labels(root, "0000")
    P2 = kitti.load_calib(root, "0000")
    m = np.isin(lab["type"], kitti.CLASSES) & (lab["trunc"] == 0) & (lab["occ"] == 0)
    ious = []
    for i in np.where(m)[0]:
        uv = kitti.project(kitti.box_corners(lab["dim"][i], lab["loc"][i], lab["ry"][i]), P2)
        ious.append(iou([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], lab["box"][i])[0, 0])
    assert np.median(ious) > 0.9


def test_depth_in_camera2_is_label_z_plus_tiny_offset(root):
    lab = kitti.load_labels(root, "0000")
    P2 = kitti.load_calib(root, "0000")
    z2 = kitti.to_camera2(lab["loc"], P2)[:, 2]
    assert np.allclose(z2 - lab["loc"][:, 2], kitti.intrinsics(P2)["t"][2]) and abs(kitti.intrinsics(P2)["t"][2]) < 0.01
