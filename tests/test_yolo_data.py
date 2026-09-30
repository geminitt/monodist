import numpy as np

from monodist import kitti, yolo_data


def test_yolo_lines_normalize_and_clip():
    lines = yolo_data.yolo_lines(np.array([[-10, 0, 100, 50], [5, 5, 5.5, 6]]), [1, 0], 200, 100)
    assert lines == ["1 0.250000 0.250000 0.500000 0.500000"]  # clipped to x in [0, 100]; the sub-pixel box is dropped


def test_build_small(root, tmp_path):
    if not kitti.image_path(root, "0012", 2).exists():
        import pytest
        pytest.skip("images of sequence 0012 not downloaded")
    yaml = yolo_data.build(root, tmp_path, splits=("test",), frames=3, only=["0012"])
    imgs = sorted((tmp_path / "images" / "test").iterdir())
    assert [p.name for p in imgs] == ["0012_000000.png", "0012_000001.png", "0012_000002.png"]
    assert all(p.resolve().exists() for p in imgs)
    lab = kitti.load_labels(root, "0012")
    n0 = int(((lab["frame"] == 0) & np.isin(lab["type"], kitti.CLASSES)).sum())
    lines = (tmp_path / "labels" / "test" / "0012_000000.txt").read_text().splitlines()
    assert len(lines) == n0
    assert all(0 <= float(v) <= 1 for line in lines for v in line.split()[1:])
    assert "names:" in yaml.read_text()


def test_extra_classes_and_dontcare_mask(root, tmp_path):
    if not kitti.image_path(root, "0012", 0).exists():
        import pytest
        pytest.skip("images of sequence 0012 not downloaded")
    classes = kitti.CLASSES + kitti.EXTRA_CLASSES
    yolo_data.build(root, tmp_path, splits=("test",), frames=1, only=["0012"], classes=classes, mask_dontcare=True)
    img = tmp_path / "images" / "test" / "0012_000000.png"
    assert not img.is_symlink()                               # frame 0 has a DontCare region, so it is a masked copy
    lab = kitti.load_labels(root, "0012")
    x1, y1, x2, y2 = lab["box"][(lab["frame"] == 0) & (lab["type"] == "DontCare")][0]
    import cv2
    pixels = cv2.imread(str(img))[int(y1) + 1:int(y2), int(x1) + 1:int(x2)]
    assert (pixels == yolo_data.GRAY).all()
    ids = [int(l.split()[0]) for l in (tmp_path / "labels" / "test" / "0012_000000.txt").read_text().splitlines()]
    assert classes.index("Cyclist") in ids                   # the cyclist of frame 0 is now labeled
    assert "3: Cyclist" in (tmp_path / "kitti.yaml").read_text()


def test_masked_leaves_the_rest_untouched():
    img = np.zeros((10, 10, 3), np.uint8)
    out = yolo_data.masked(img, [[2, 2, 4, 4]])
    assert (out[2:5, 2:5] == yolo_data.GRAY).all() and out.sum() == 9 * 3 * yolo_data.GRAY
    assert img.sum() == 0
