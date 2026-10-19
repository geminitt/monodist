import numpy as np

from monodist import calibrate, kitti, mlp, temporal
from test_evaluate import oracle_detections


def test_calibration_never_opens_the_held_out_test(root, tmp_path, monkeypatch):
    oracle_detections(root, tmp_path / "oracle", seqs=kitti.DEVELOPMENT)
    load = kitti.load_labels

    def guarded(root, seq):
        assert seq not in kitti.TEST
        return load(root, seq)

    monkeypatch.setattr(kitti, "load_labels", guarded)
    monkeypatch.setattr(temporal, "MEAS_REL", (0.02,))
    monkeypatch.setattr(temporal, "ACCEL", (1.,))
    meta = calibrate.fit(root, tmp_path / "oracle", tmp_path / "out", epochs=2)
    _, saved = mlp.load(tmp_path / "out" / "distance.pt")
    assert saved == meta and meta["calibration_only"] is True
    assert "test" not in meta["split"] and meta["temporal"] == {"meas_rel": 0.02, "accel": 1.}
    assert set(np.load(tmp_path / "out" / "validation.npz")["seq"]) == set(kitti.SPLIT["val"])
