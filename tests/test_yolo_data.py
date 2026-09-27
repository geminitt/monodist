import numpy as np

from monodist import kitti, yolo_data


def test_yolo_lines_normalise_and_clip():
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
