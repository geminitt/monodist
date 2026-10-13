import numpy as np

from monodist import evaluate, kitti


def oracle_detections(root, out, jitter=0.0, seed=0):
    """Labeled boxes of every sequence written as if a detector had produced them."""
    rng = np.random.default_rng(seed)
    out.mkdir()
    for seq in kitti.SEQUENCES:
        lab = kitti.load_labels(root, seq)
        m = np.isin(lab["type"], kitti.CLASSES)
        box = lab["box"][m] + rng.normal(0, jitter, (m.sum(), 4))
        hw = (370, 1224) if seq in ("0014", "0015", "0016", "0017") else (375, 1242)
        np.savez(out / f"{seq}.npz", frame=lab["frame"][m], box=box, score=np.full(m.sum(), 0.9),
                 cls=np.array([kitti.CLASSES.index(t) for t in lab["type"][m]]), image_hw=np.array(hw))


def test_perfect_detector_end_to_end(root, tmp_path):
    oracle_detections(root, tmp_path / "oracle")
    const = evaluate.constants(root)
    assert 1.4 < const["class_height"][0] < 1.7 and 1.6 < const["class_height"][1] < 1.9
    assert 1.4 < const["cam_height"] < 1.8
    res, rows = evaluate.run(root, tmp_path / "oracle", n_boot=20, mlp_epochs=15)
    assert res["constants"] == const
    assert res["detection"]["mAP50-95"] > 0.999
    rec = res["recall_by_distance"]["Car"]
    assert sum(rec["found"]) == sum(rec["total"]) > 0
    d, floor = res["distance"], res["distance_on_label_boxes"]
    # detector boxes are the label boxes, so the geometric methods must score exactly their floor
    assert np.isclose(d["size"]["all"]["absrel_mean"][0], floor["size"]["all"]["absrel_mean"][0])
    assert np.isclose(d["ground"]["all"]["absrel_mean"][0], floor["ground"]["all"]["absrel_mean"][0])
    assert d["size"]["all"]["n"] == len(rows["z"])
    assert 0 <= res["ground_fallback_share"] < 0.5
    assert all(v > 0.999 for v in res["box_iou_median"].values())
    assert d["mlp"]["all"]["absrel_mean"][0] < 0.5


def test_compare_on_common_objects(root, tmp_path):
    oracle_detections(root, tmp_path / "a")
    oracle_detections(root, tmp_path / "b", jitter=2.0)
    _, ra = evaluate.run(root, tmp_path / "a", n_boot=20, mlp_epochs=15)
    _, rb = evaluate.run(root, tmp_path / "b", n_boot=20, mlp_epochs=15)
    c = evaluate.compare(ra, rb, n_boot=20)
    assert c["n_common"] > 0.8 * len(ra["z"])
    assert c["size"]["b - a"][0] > 0  # jittered boxes give worse known-size distances


def test_summary_of_one_class_only():
    """A report whose rows hold cars but no pedestrian: the pedestrian entries are empty, not an error."""
    one = lambda v: np.array([v])  # noqa: E731
    rows = {"seq": one("0000"), "track": one(0), "cls": one(0), "trunc": one(0), "z": one(10.), "fallback": one(False),
            **{f"{p}_{m}": one(11.) for p in ("pred", "label") for m in evaluate.METHODS}}
    s = evaluate.summarize(rows, n_boot=10)
    assert s["distance"]["size"]["Pedestrian"] == {"n": 0, "tracks": 0}
    assert s["distance"]["paired_absrel_difference"]["size - ground"]["Pedestrian"] is None
    assert np.isclose(s["distance"]["size"]["Car"]["absrel_mean"][0], 0.1)
