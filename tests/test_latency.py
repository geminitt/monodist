import json

import pytest

from monodist import latency


def test_existing_timing_is_reused_only_for_the_same_configuration(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(latency, "run", lambda *a: calls.append(a) or {"total": {"median_ms": [1.0]}, "fps_per_round": [1.0]})
    monkeypatch.setattr(latency, "STAGES", ("total",))
    monkeypatch.setattr(latency, "data_identity", lambda *a: {"image": "sha256"})
    w, out, distance = tmp_path / "best.pt", tmp_path / "t.json", tmp_path / "distance.pt"
    w.write_bytes(b"weights")
    distance.write_bytes(b"trained ensemble")
    argv = ["--weights", str(w), "--distance", str(distance), "--root", "kitti", "--out", str(out)]
    latency.main(argv)
    assert len(calls) == 1 and json.loads(out.read_text())["signature"]["weights"]["name"] == "best.pt"
    latency.main(argv)  # same weights and settings: skipped
    assert len(calls) == 1
    with pytest.raises(SystemExit, match="another configuration"):
        latency.main(argv + ["--half"])
    distance.write_bytes(b"different trained ensemble")
    with pytest.raises(SystemExit, match="another configuration"):
        latency.main(argv)
    w.write_bytes(b"retrained weights")
    with pytest.raises(SystemExit, match="another configuration"):
        latency.main(argv)


def test_surrogate_latency_is_refused():
    with pytest.raises(ValueError, match="saved trained distance"):
        latency.run("weights", "root", "0012", 5, 1, 1, 1280, False)


def test_wrong_detector_pairing_is_refused_before_inference(tmp_path, monkeypatch):
    w = tmp_path / "best.pt"
    w.write_bytes(b"different detector")
    monkeypatch.setattr(latency.mlp, "load", lambda p: ([], {
        "temporal": {"meas_rel": 0.02, "accel": 1.},
        "detector": {"weights": {"sha256": "0" * 64}}}))
    with pytest.raises(ValueError, match="weights differ"):
        latency.run(w, "root", "0012", 3, 1, 1, 1280, False, distance="distance.pt")


def test_timing_from_other_hardware_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(latency, "data_identity", lambda *a: {})
    monkeypatch.setattr(latency, "hardware_identity", lambda: {"gpus": ["T4"]})
    monkeypatch.setattr(latency, "STAGES", ("total",))
    monkeypatch.setattr(latency, "run", lambda *a: {"total": {"median_ms": [1.]}, "fps_per_round": [1.]})
    w, d, out = tmp_path / "weights.pt", tmp_path / "distance.pt", tmp_path / "timing.json"
    w.write_bytes(b"detector")
    d.write_bytes(b"distance")
    argv = ["--weights", str(w), "--distance", str(d), "--root", "data", "--out", str(out)]
    latency.main(argv)
    monkeypatch.setattr(latency, "hardware_identity", lambda: {"gpus": []})
    with pytest.raises(SystemExit, match="another configuration"):
        latency.main(argv)
