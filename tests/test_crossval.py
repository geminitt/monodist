import numpy as np

from monodist import crossval, kitti
from test_evaluate import oracle_detections


def test_cross_validation_covers_every_development_sequence_once(root, tmp_path):
    oracle_detections(root, tmp_path / "oracle", seqs=kitti.DEVELOPMENT)  # no detections on the held-out TEST
    report, rows = crossval.cross_validate(root, [tmp_path / "oracle"] * 3, n_boot=20, mlp_epochs=5)
    assert sorted(np.unique(rows["seq"])) == kitti.DEVELOPMENT
    assert len(np.unique(rows["keys"])) == len(rows["keys"])
    assert report["detection"]["mAP50-95"]["mean"] > 0.999 and len(report["detection"]["mAP50-95"]["per_fold"]) == 3
    for c in ("Car", "Van", "LargeVehicle", "Pedestrian", "Cyclist", "Vehicle group", "object"):
        assert sum(report["recall_by_distance"][c]["found"]) == sum(report["recall_by_distance"][c]["total"]) > 0
    assert report["detection"]["agnostic AP50-95"]["mean"] > 0.999


def test_selection_prefers_the_better_detector_on_val(root, tmp_path):
    for k in range(3):
        oracle_detections(root, tmp_path / f"f{k}_a_nms", seqs=kitti.DEVELOPMENT)
        oracle_detections(root, tmp_path / f"f{k}_b_nms", jitter=3.0, seqs=kitti.DEVELOPMENT)
    table, best = crossval.select(root, tmp_path, ["a", "b"])
    assert best == {0: "a", 1: "a", 2: "a"} and all(t["a"] > t["b"] for t in table.values())
