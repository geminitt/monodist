import numpy as np

from monodist import kitti, yolo_data


def test_yolo_lines_normalize_and_clip():
    lines = yolo_data.yolo_lines(np.array([[-10, 0, 100, 50], [5, 5, 5.5, 6]]), [1, 0], 200, 100)
    assert lines == ["1 0.250000 0.250000 0.500000 0.500000"]  # clipped to x in [0, 100]; the sub-pixel box is dropped


def test_build_small(root, tmp_path):
    if not kitti.image_path(root, "0012", 2).exists():
        import pytest
        pytest.skip("images of sequence 0012 not downloaded")
    yaml = yolo_data.build(root, tmp_path, split={"test": ["0012"]}, splits=("test",), frames=3)
    imgs = sorted((tmp_path / "images" / "test").iterdir())
    assert [p.name for p in imgs] == ["0012_000000.png", "0012_000001.png", "0012_000002.png"]
    assert all(p.resolve().exists() for p in imgs)
    lab = kitti.load_labels(root, "0012")
    n0 = int(((lab["frame"] == 0) & (kitti.classify(lab["type"]) >= 0)).sum())
    lines = (tmp_path / "labels" / "test" / "0012_000000.txt").read_text().splitlines()
    assert len(lines) == n0
    assert all(0 <= float(v) <= 1 for line in lines for v in line.split()[1:])
    assert "names:" in yaml.read_text() and "4: Cyclist" in yaml.read_text()
    ids = {int(line.split()[0]) for line in lines}
    assert 4 in ids  # the cyclist of frame 0 is labeled with its own class


def test_dontcare_mask_spares_labeled_boxes():
    boxes = np.array([[2.5, 1.0, 8.2, 6.0], [4.0, 2.0, 6.0, 4.0], [-3.0, 0.0, 1.0, 2.0]])
    types = np.array(["DontCare", "Pedestrian", "DontCare"])
    mask = yolo_data.dontcare_mask(boxes, types, 10, 8)
    assert mask[1:6, 2:9].sum() == 5 * 7 - 2 * 2  # DontCare pixels (floor/ceil) minus the labeled box
    assert not mask[2:4, 4:6].any() and mask[0:2, 0:1].all()  # the box cut by the image edge is clipped
    assert mask.sum() == 31 + 2


def test_build_with_masked_dontcare(root, tmp_path):
    if not kitti.image_path(root, "0012", 2).exists():
        import pytest
        pytest.skip("images of sequence 0012 not downloaded")
    from PIL import Image
    lab = kitti.load_labels(root, "0012")
    dc = sorted({int(f) for f in lab["frame"][lab["type"] == "DontCare"] if f < 10})
    assert dc, "sequence 0012 has DontCare regions in its first frames"
    fr = dc[0]
    n = yolo_data.mask_images(root, ["0012"], tmp_path / "masked", frames=fr + 1)
    yolo_data.build(root, tmp_path / "data", split={"train": ["0012"]}, splits=("train",), frames=fr + 1,
                    masked=tmp_path / "masked")
    assert n == len(dc[:dc.index(fr) + 1])
    link = tmp_path / "data" / "images" / "train" / f"0012_{fr:06d}.png"
    assert link.resolve().parent == (tmp_path / "masked").resolve()
    orig = np.array(Image.open(kitti.image_path(root, "0012", fr)).convert("RGB"))
    new = np.array(Image.open(link))
    m = lab["frame"] == fr
    mask = yolo_data.dontcare_mask(lab["box"][m], lab["type"][m], orig.shape[1], orig.shape[0])
    assert mask.any() and np.all(new[mask] == yolo_data.FILL) and np.array_equal(new[~mask], orig[~mask])
    # labels are unchanged by masking, and a rebuild in the same folder follows the new setting
    plain = yolo_data.build(root, tmp_path / "plain", split={"train": ["0012"]}, splits=("train",), frames=fr + 1)
    yolo_data.build(root, tmp_path / "data", split={"train": ["0012"]}, splits=("train",), frames=fr + 1)
    assert link.resolve() == kitti.image_path(root, "0012", fr).resolve()
    assert (plain.parent / "labels" / "train" / f"0012_{fr:06d}.txt").read_text() == \
        (tmp_path / "data" / "labels" / "train" / f"0012_{fr:06d}.txt").read_text()
