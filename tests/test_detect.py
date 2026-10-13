import json

import numpy as np
import pytest

from monodist import detect


def write(path):
    np.savez(path, frame=np.array([0]), box=np.array([[0., 0., 10., 10.]]), score=np.array([0.9]), cls=np.array([0]),
             image_hw=np.array([375, 1242]))


def test_merge_skips_unfinished_temporary_files(tmp_path):
    write(tmp_path / "0000.npz")
    write(tmp_path / "0001.tmp.npz")  # left by a run stopped between writing and renaming
    det, sizes = detect.merge(tmp_path)
    assert list(sizes) == ["0000"] and len(det["score"]) == 1


def test_weights_identity_hashes_the_file(tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights")
    ident = detect.weights_identity(w)
    assert ident["name"] == "best.pt" and len(ident["sha256"]) == 64
    w.write_bytes(b"other weights")
    assert detect.weights_identity(w)["sha256"] != ident["sha256"]


def test_signature_is_written_then_enforced(tmp_path):
    sig = {"weights": {"name": "best.pt", "sha256": "0" * 64}, "imgsz": 1280}
    detect.check_signature(tmp_path, sig)
    assert json.loads((tmp_path / "run.json").read_text()) == sig
    detect.check_signature(tmp_path, sig)  # same configuration: resume
    with pytest.raises(SystemExit, match="another configuration"):
        detect.check_signature(tmp_path, {**sig, "imgsz": 640})


def test_unsigned_detections_are_not_resumed(tmp_path):
    write(tmp_path / "0000.npz")  # written before signatures existed: its configuration is unknown
    with pytest.raises(SystemExit, match="no run.json"):
        detect.check_signature(tmp_path, {"imgsz": 1280})
