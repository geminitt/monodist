import json

import numpy as np

from monodist import analysis, kitti, temporal
from test_evaluate import oracle_detections


def test_fold_artifacts_and_filter_selection_use_validation_only(root, tmp_path, monkeypatch):
    oracle_detections(root, tmp_path / "oracle", seqs=kitti.DEVELOPMENT)
    out = tmp_path / "analysis"
    out.mkdir()
    fold = kitti.FOLDS[0]
    original = temporal.select_filter
    monkeypatch.setattr(temporal, "MEAS_REL", (0.01,))
    monkeypatch.setattr(temporal, "ACCEL", (1.,))

    def guarded_selection(rows, ids, speed):
        assert set(rows["seq"]) == set(fold["val"])
        assert not set(rows["seq"]) & set(fold["test"])
        return original(rows, ids, speed)

    monkeypatch.setattr(temporal, "select_filter", guarded_selection)
    analysis.fold_analysis(root, tmp_path / "oracle", fold, out, epochs=3)
    report = json.loads((out / "temporal.json").read_text())
    scored = dict(np.load(out / "tracked_test.npz"))
    assert set(scored["seq"]) == set(fold["test"])
    assert report["n"] > 100 and report["selection"]["sequences"] == fold["val"]
    assert report["matched_at_least_2m"] <= report["detections"]
    assert report["association"]["unmatched_detections"] == 0
    assert (out / "distance.pt").is_file() and (out / "diagnostics.json").is_file()
    assert set(json.loads((out / "thresholds_validation.json").read_text())) == set(kitti.CLASSES)
