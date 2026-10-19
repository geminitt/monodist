import numpy as np
import torch

from monodist import geometry, mlp, pipeline


def test_shared_postprocessing_uses_five_classes_thresholds_and_resets_tracking(monkeypatch):
    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)
    model = mlp.DistanceMLP(len(geometry.FEATURES))
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.net[-1].bias.fill_(np.log(10.))
    pipe.models = [model] * 5
    pipe.conf = np.array([0.5, 0.5, 0.5, 0.5, 0.3])
    pipe.temporal_settings = {"meas_rel": 0.02, "accel": 1.}
    pipe.reset()
    raw = {"box": np.array([[100., 100., 120., 140.], [200., 100., 220., 140.], [300., 100., 320., 140.]]),
           "cls": np.array([0, 4, 2]), "score": np.array([0.9, 0.4, 0.4]), "image_hw": (375, 1242)}
    first = pipe.finish(raw, (700., 600., 180.))
    second = pipe.finish(raw, (700., 600., 180.))
    assert first["cls"].tolist() == [0, 4]  # Cyclist uses its own threshold; LargeVehicle is discarded
    assert np.allclose(first["z"], 10.) and (first["range"] >= first["z"]).all()
    assert np.array_equal(first["track"], second["track"]) and np.isinf(second["ttc"]).all()
    assert pipe.frame == 2
    pipe.reset()
    assert pipe.frame == 0 and pipe.tracker.tracks == {}
    pipe.detector, pipe.imgsz, pipe.nms, pipe.coco, pipe.half = None, 1280, True, False, False
    calls = []
    monkeypatch.setattr(pipeline.detect, "predict_images", lambda *args: calls.append(args) or [raw])
    out = pipe(["image"], (700., 600., 180.))
    assert calls[0][3] == 0.3 and out[0]["cls"].tolist() == [0, 4]
