import numpy as np

from monodist import evaluate, kitti


def oracle_detections(root, out, jitter=0.0, seed=0, seqs=kitti.SEQUENCES):
    """Labeled boxes of the class groups written as if a detector had produced them."""
    rng = np.random.default_rng(seed)
    out.mkdir()
    for seq in seqs:
        lab = kitti.load_labels(root, seq)
        cls = kitti.classify(lab["type"])
        m = cls >= 0
        box = lab["box"][m] + rng.normal(0, jitter, (m.sum(), 4))
        hw = (370, 1224) if seq in ("0014", "0015", "0016", "0017") else (375, 1242)
        np.savez(out / f"{seq}.npz", frame=lab["frame"][m], box=box, score=np.full(m.sum(), 0.9),
                 cls=cls[m], image_hw=np.array(hw))


def test_perfect_detector_end_to_end(root, tmp_path):
    oracle_detections(root, tmp_path / "oracle")
    const = evaluate.constants(root)
    car, van, large, ped, cyc = const["class_height"]
    assert 1.4 < car < 1.7 and 1.8 < van < 2.5 and large > 2.5 and 1.5 < ped < 1.9 and 1.5 < cyc < 1.9
    assert 1.4 < const["cam_height"] < 1.8
    res, rows = evaluate.run(root, tmp_path / "oracle", n_boot=20, mlp_epochs=15)
    assert res["constants"] == const
    assert res["detection"]["mAP50-95"] > 0.999
    for c in ("Car", "Van", "LargeVehicle", "Pedestrian", "Cyclist"):
        rec = res["recall_by_distance"][c]
        assert sum(rec["found"]) == sum(rec["total"]) > 0
    for g in ("Vehicle", "Person", "Cyclist"):
        assert res["detection_groups"][g]["AP50-95"] > 0.999
    rec = res["recall_by_distance_agnostic"]
    assert sum(rec["found"]) == sum(rec["total"]) > 0
    assert res["detection_agnostic"]["mAP50-95"] > 0.999
    prec = res["precision_by_predicted_distance"]
    assert sum(prec["hits"]) == sum(prec["total"]) > 0  # every kept detection is a labeled object
    assert np.all(rows["range"] >= rows["z"])
    d, floor = res["distance"], res["distance_on_label_boxes"]
    # detector boxes are the label boxes, so the geometric methods must score exactly their floor
    assert np.isclose(d["size"]["all"]["absrel_mean"][0], floor["size"]["all"]["absrel_mean"][0])
    assert np.isclose(d["ground"]["all"]["absrel_mean"][0], floor["ground"]["all"]["absrel_mean"][0])
    assert d["size"]["all"]["n"] == len(rows["z"])
    assert 0 <= res["ground_fallback_share"] < 0.5
    assert all(v > 0.999 for v in res["box_iou_median"].values())
    assert d["mlp"]["all"]["absrel_mean"][0] < 0.5
    assert res["range"]["mlp"]["all"]["n"] == d["mlp"]["all"]["n"]
    assert d["mlp"]["Vehicle"]["n"] == sum(d["mlp"][c]["n"] for c in ("Car", "Van", "LargeVehicle"))


def test_a_fold_reads_only_its_own_sequences(root, tmp_path):
    """Detections exist only for the development sequences: a fold must not need the held-out ones."""
    oracle_detections(root, tmp_path / "dev", seqs=kitti.DEVELOPMENT)
    info, rows = evaluate.predict(root, tmp_path / "dev", kitti.FOLDS[0], mlp_epochs=3)
    assert set(np.unique(rows["seq"])) == set(kitti.FOLDS[0]["test"])


def test_tracking_artifacts_do_not_read_held_out_files(root, tmp_path, monkeypatch):
    """Guard both detection and label reads, not just the final set of reported rows."""
    from monodist import mlp
    oracle_detections(root, tmp_path / "dev", seqs=kitti.DEVELOPMENT)
    original_merge, original_labels = evaluate.merge, kitti.load_labels

    def guarded_merge(directory, seqs=None):
        assert seqs is not None and not set(seqs) & set(kitti.TEST)
        return original_merge(directory, seqs)

    def guarded_labels(root, seq):
        assert seq not in kitti.TEST
        return original_labels(root, seq)

    monkeypatch.setattr(evaluate, "merge", guarded_merge)
    monkeypatch.setattr(kitti, "load_labels", guarded_labels)
    fold = kitti.FOLDS[0]
    info, rows = evaluate.predict(root, tmp_path / "dev", fold, mlp_epochs=3,
                                  save_model=tmp_path / "distance.pt", save_observations=tmp_path / "obs.npz")
    obs = dict(np.load(tmp_path / "obs.npz"))
    assert set(obs["seq"]) == set(fold["val"] + fold["test"])
    assert np.all(obs["track"] >= 0)  # an oracle has no false positives
    assert np.all(obs["status"] == 1)
    model, meta = mlp.load(tmp_path / "distance.pt")
    assert meta["conf"] == info["conf"]
    cam = evaluate.camera(root, sorted(set(obs["seq"])),
                          {s: (370, 1224) if s in ("0014", "0015") else (375, 1242) for s in set(obs["seq"])})
    assert np.allclose(mlp.predict(model, evaluate.feats(obs, cam)), obs["pred_mlp"])
    assert np.allclose(rows["label_mlp"], rows["pred_mlp"])


def test_compare_on_common_objects(root, tmp_path):
    oracle_detections(root, tmp_path / "a")
    oracle_detections(root, tmp_path / "b", jitter=2.0)
    _, ra = evaluate.run(root, tmp_path / "a", n_boot=20, mlp_epochs=15)
    _, rb = evaluate.run(root, tmp_path / "b", n_boot=20, mlp_epochs=15)
    c = evaluate.compare(ra, rb, n_boot=20)
    assert c["n_common"] > 0.8 * len(ra["z"])
    assert c["size"]["b - a"][0] > 0  # jittered boxes give worse known-size distances


def test_summary_of_one_class_only():
    """A report whose rows hold cars only: the other classes' entries are empty, not an error."""
    one = lambda v: np.array([v])  # noqa: E731
    rows = {"seq": one("0000"), "track": one(0), "cls": one(0), "trunc": one(0), "z": one(10.), "range": one(10.),
            "fallback": one(False), **{f"{p}_{m}": one(11.) for p in ("pred", "label", "range") for m in evaluate.METHODS}}
    s = evaluate.summarize(rows, n_boot=10)
    assert s["distance"]["size"]["Pedestrian"] == {"n": 0, "tracks": 0}
    assert s["distance"]["paired_absrel_difference"]["size - ground"]["Cyclist"] is None
    assert np.isclose(s["distance"]["size"]["Car"]["absrel_mean"][0], 0.1)
    assert np.isclose(s["distance"]["size"]["Vehicle"]["absrel_mean"][0], 0.1)
