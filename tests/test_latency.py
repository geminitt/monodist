import json

import pytest

from monodist import latency


def test_existing_timing_is_reused_only_for_the_same_configuration(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(latency, "run", lambda *a: calls.append(a) or {"total": {"median_ms": [1.0]}, "fps_per_round": [1.0]})
    monkeypatch.setattr(latency, "STAGES", ("total",))
    w, out = tmp_path / "best.pt", tmp_path / "t.json"
    w.write_bytes(b"weights")
    argv = ["--weights", str(w), "--root", "kitti", "--out", str(out)]
    latency.main(argv)
    assert len(calls) == 1 and json.loads(out.read_text())["signature"]["weights"]["name"] == "best.pt"
    latency.main(argv)  # same weights and settings: skipped
    assert len(calls) == 1
    with pytest.raises(SystemExit, match="another configuration"):
        latency.main(argv + ["--half"])
    w.write_bytes(b"retrained weights")
    with pytest.raises(SystemExit, match="another configuration"):
        latency.main(argv)
