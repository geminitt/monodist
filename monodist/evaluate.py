"""Evaluate a detector and the three distance methods on the test sequences of a split; one JSON per run.

    python -m monodist.evaluate --root data/kitti_tracking --det results/det/coco_1280 results/det/ft_1280

Everything here runs on CPU from saved detections: fitting the geometric constants on the train labels,
training the distance MLP on this detector's boxes on the train sequences, choosing the score threshold on
the val sequences, and scoring on the test sequences. `predict` does one split and returns per-object rows;
`summarize` turns rows (possibly pooled over the folds of a cross-validation) into the reported numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry, kitti, match, metrics, mlp
from .detect import merge

MIN_Z = 2.0  # closer objects are all cut by the image border; AbsRel near Z = 0 would dominate the mean
METHODS = ("size", "ground", "mlp")


def constants(root, train=None):
    """Known-size heights per class group and the camera height above the road, from the train labels only.

    The camera height is the median bottom of the Car boxes (cars stand on the road; vans and trucks too, but
    the Car rows alone are plenty and their bottoms are the most reliable).
    """
    heights = {c: [] for c in range(len(kitti.CLASSES))}
    road = []
    for seq in train or kitti.SPLIT["train"]:
        lab = kitti.load_labels(root, seq)
        y = kitti.to_camera2(lab["loc"], kitti.load_calib(root, seq))[:, 1]
        cls = kitti.classify(lab["type"])
        for c in heights:
            heights[c].extend(lab["dim"][cls == c, 0])
        road.extend(y[lab["type"] == "Car"])
    return {"class_height": [float(np.mean(heights[c])) for c in heights], "cam_height": float(np.median(road))}


def camera(root, seqs, sizes):
    cam = {}
    for s in seqs:
        k = kitti.intrinsics(kitti.load_calib(root, s))
        h, w = sizes[s]
        cam[s] = (k["f"], k["cx"], k["cy"], w, h)
    return cam


def per_row(cam, seq, i):
    return np.array([cam[s][i] for s in seq])


def geometric(pairs, box, cam, const):
    f, cy, h = per_row(cam, pairs["seq"], 0), per_row(cam, pairs["seq"], 2), per_row(cam, pairs["seq"], 4)
    za = geometry.size_distance(box, pairs["cls"], f, const["class_height"])
    zb, fallback = geometry.ground_or_size(box, pairs["cls"], f, cy, const["cam_height"], h, const["class_height"])
    return za, zb, fallback


def to_range(rows, z, cam):
    s = rows["seq"]
    return geometry.depth_to_range(rows["box"], z, per_row(cam, s, 0), per_row(cam, s, 1), per_row(cam, s, 2))


def precision_by_distance(det, status, z_pred, bins):
    """Share of kept detections that hit an object (IoU >= 0.5, any class), per bin of the predicted depth.

    This is what the decision layer sees: every kept detection is an obstacle at its predicted distance.
    """
    keep = status[:, 0] != match.IGNORED
    b = np.searchsorted(bins, z_pred, side="right") - 1
    hits, total = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for k in range(len(bins) - 1):
        m = keep & (b == k)
        total[k], hits[k] = m.sum(), (status[m, 0] == match.TP).sum()
    return {"hits": hits.tolist(), "total": total.tolist()}


def feats(pairs, cam):
    s = pairs["seq"]
    return geometry.features(pairs["box"], pairs["cls"], per_row(cam, s, 0), per_row(cam, s, 1), per_row(cam, s, 2),
                             per_row(cam, s, 3), per_row(cam, s, 4))


def subset(d, seqs):
    m = np.isin(d["seq"], seqs)
    return {k: v[m] for k, v in d.items()}


def observations(det, gt, model, cam):
    """Kept detections, including false positives, for tracking without label-based association.

    Labels are attached with class-agnostic IoU >= 0.5 matching for scoring only. They must never be inputs
    to the tracker. Matching is repeated after thresholding so discarded detections cannot claim an object.
    """
    ad, ag = match.agnostic(det, gt)
    status, idx = match.evaluate(ad, ag, thresholds=[0.5])
    out = {k: det[k] for k in ("seq", "frame", "box", "score", "cls")}
    n = len(det["score"])
    out.update(pred_mlp=mlp.predict(model, feats(det, cam)), status=status[:, 0],
               track=np.full(n, -1, int), z=np.full(n, np.nan), gt_cls=np.full(n, -1, int))
    for i in np.flatnonzero(status[:, 0] == match.TP):
        g = gt[(det["seq"][i], det["frame"][i])]
        j = idx[i, 0]
        out["track"][i], out["z"][i], out["gt_cls"][i] = g["track"][j], g["z"][j], g["cls"][j]
    return out


def predict(root, det_dir, split=None, seed=0, mlp_epochs=150, save_model=None, save_observations=None):
    """One split: detection metrics on test, and every matched test object with the three distance predictions.

    save_model: a path where the distance ensemble is saved with its thresholds, constants and provenance
    (monodist.pipeline loads it).
    """
    split = split or kitti.SPLIT
    # Only the split's own sequences are read: a cross-validation fold never touches the held-out kitti.TEST.
    seqs = sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))
    det, sizes = merge(det_dir, seqs=seqs)
    missing = sorted(set(seqs) - set(sizes))
    assert not missing, f"{det_dir}: no detections for sequences {missing}"
    det = subset(det, seqs)
    cam = camera(root, seqs, sizes)
    const = constants(root, split["train"])
    gt = match.ground_truth(root, seqs)
    status, gt_idx = match.evaluate(det, gt)
    test_gt = {k: v for k, v in gt.items() if k[0] in split["test"]}
    tm = np.isin(det["seq"], split["test"])
    tdet = {k: v[tm] for k, v in det.items()}
    # Operating point: per class, the score threshold with the best F1 on the val sequences. A fixed 0.25 would
    # favor whichever detector happens to output higher scores; fine-tuning changes the score scale.
    val_gt = {k: v for k, v in gt.items() if k[0] in split["val"]}
    vm = np.isin(det["seq"], split["val"])
    conf = [match.f1_threshold(det["score"][vm & (det["cls"] == c)], status[vm & (det["cls"] == c), 0],
                               int(sum((g["cls"] == c).sum() for g in val_gt.values()))) for c in range(len(kitti.CLASSES))]
    info = {"det_dir": str(det_dir), "split": split, "conf": dict(zip(kitti.CLASSES, conf)), "min_z": MIN_Z,
            "constants": const, "detection": match.detection_report(tdet, test_gt, status[tm]),
            "recall_by_distance": match.recall_by_distance(tdet, test_gt, status[tm], gt_idx[tm],
                                                           match.per_detection_conf(tdet, conf), metrics.BINS)}
    info["val_detection"] = match.detection_report({k: v[vm] for k, v in det.items()}, val_gt, status[vm])
    # Reporting groups (Vehicle = Car + Van + LargeVehicle): a van found as a car counts as found.
    gdet, ggt = match.regroup(tdet, test_gt, kitti.GROUP_OF)
    gstatus, _ = match.evaluate(gdet, ggt)
    info["detection_groups"] = match.detection_report(gdet, ggt, gstatus, names=kitti.GROUPS)
    # Class-agnostic view (the decision layer's): was the object found at all, whatever class was predicted?
    # Each detection keeps the threshold of the class it was predicted as.
    adet, agt = match.agnostic(tdet, test_gt)
    astatus, _ = match.evaluate(adet, agt)
    info["detection_agnostic"] = match.detection_report(adet, agt, astatus, names=("object",))
    # Recall and precision at the operating point: classes in one group have different thresholds, so matching
    # is repeated on the kept detections only; a discarded detection must not take an object from a kept one.
    kept = tdet["score"] >= match.per_detection_conf(tdet, conf)
    kdet = {k: v[kept] for k, v in tdet.items()}
    every = np.zeros(int(kept.sum()))  # the thresholds are already applied
    gdet, ggt = match.regroup(kdet, test_gt, kitti.GROUP_OF)
    gstatus, ggt_idx = match.evaluate(gdet, ggt, thresholds=[0.5])
    info["recall_by_distance_groups"] = match.recall_by_distance(
        gdet, ggt, gstatus, ggt_idx, every, metrics.BINS, names=kitti.GROUPS)
    adet, agt = match.agnostic(kdet, test_gt)
    kstatus, agt_idx = match.evaluate(adet, agt, thresholds=[0.5])
    info["recall_by_distance_agnostic"] = match.recall_by_distance(
        adet, agt, kstatus, agt_idx, every, metrics.BINS, names=("object",))["object"]

    pairs = match.matched_pairs(det, gt, status, gt_idx, match.per_detection_conf(det, conf))
    near = pairs["z"] < MIN_Z
    info["excluded_below_min_z"] = int(near[np.isin(pairs["seq"], split["test"])].sum())
    pairs = {k: v[~near] for k, v in pairs.items()}
    train, test = subset(pairs, split["train"]), subset(pairs, split["test"])
    # How tight this detector's boxes are on each part: a fine-tuned detector is tighter on images it trained on,
    # which would make an MLP trained on those boxes trust the box height too much.
    info["box_iou_median"] = {part: float(np.median(geometry.iou_pairs(p["box"], p["gt_box"])))
                              for part, p in ((part, subset(pairs, split[part])) for part in ("train", "val", "test"))}
    model, epochs, curve = mlp.train(feats(train, cam), train["z"], train["seq"], max_epochs=mlp_epochs, seed=seed)
    info["mlp"] = {"epochs": epochs, "heldout_l1_log": [float(v) for v in curve], "n_train": int(len(train["z"]))}
    if save_model:
        run = Path(det_dir) / "run.json"
        mlp.save(save_model, model, {
            "features": list(geometry.FEATURES), "classes": list(kitti.CLASSES), "conf": info["conf"],
            "constants": const, "min_z": MIN_Z, "epochs": epochs, "split": split, "det_dir": str(det_dir),
            "detector": json.loads(run.read_text()) if run.exists() else None})

    if save_observations:
        seqs_observed = split["val"] + split["test"]
        od = subset(det, seqs_observed)
        keep = od["score"] >= match.per_detection_conf(od, conf)
        od = {k: v[keep] for k, v in od.items()}
        og = {k: v for k, v in gt.items() if k[0] in seqs_observed}
        np.savez_compressed(save_observations, **observations(od, og, model, cam))

    info["precision_by_predicted_distance"] = precision_by_distance(
        kdet, kstatus, mlp.predict(model, feats(kdet, cam)), metrics.BINS)

    za, zb, fallback = geometric(test, test["box"], cam, const)
    ga, gb, _ = geometric(test, test["gt_box"], cam, const)
    zm = mlp.predict(model, feats(test, cam))
    rows = {k: test[k] for k in ("seq", "frame", "track", "cls", "trunc", "occ", "z", "range", "box", "gt_box", "score")}
    rows.update({"pred_size": za, "pred_ground": zb, "pred_mlp": zm,
                 "label_size": ga, "label_ground": gb, "fallback": fallback,
                 "range_size": to_range(test, za, cam), "range_ground": to_range(test, zb, cam),
                 "range_mlp": to_range(test, zm, cam),
                 "label_mlp": mlp.predict(model, geometry.features(
                     test["gt_box"], test["cls"], per_row(cam, test["seq"], 0), per_row(cam, test["seq"], 1),
                     per_row(cam, test["seq"], 2), per_row(cam, test["seq"], 3), per_row(cam, test["seq"], 4)))})
    rows["keys"] = np.array([f"{s}/{f}/{t}" for s, f, t in zip(test["seq"], test["frame"], test["track"])])
    return info, rows


def breakdown(preds, rows, n_boot):
    """Every method, overall, per class, per class x distance bin, plus paired differences between methods."""
    z, clusters = rows["z"], np.array([f"{s}/{t}" for s, t in zip(rows["seq"], rows["track"])])
    b = metrics.distance_bin(z)
    out = {}
    # every class, then the groups that gather several classes (Vehicle)
    parts = [(cname, rows["cls"] == c) for c, cname in enumerate(kitti.CLASSES)]
    parts += [(g, kitti.GROUP_OF[rows["cls"]] == k) for k, g in enumerate(kitti.GROUPS) if (kitti.GROUP_OF == k).sum() > 1]
    for name, p in preds.items():
        out[name] = {"all": metrics.summarize(p, z, clusters, n_boot)}
        for cname, m in parts:
            out[name][cname] = metrics.summarize(p[m], z[m], clusters[m], n_boot)
            whole = m & (rows["trunc"] == 0)  # not cut by the image border (KITTI truncation level 0)
            out[name][f"{cname} not truncated"] = metrics.summarize(p[whole], z[whole], clusters[whole], n_boot)
            for k, bname in enumerate(metrics.BIN_NAMES):
                mb = m & (b == k)
                out[name][f"{cname} {bname}"] = metrics.summarize(p[mb], z[mb], clusters[mb], n_boot)
    diffs = {}
    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ea, eb = metrics.abs_rel(preds[names[i]], z), metrics.abs_rel(preds[names[j]], z)
            d = {"all": metrics.paired_difference(ea, eb, clusters, n_boot)}
            for cname, m in parts:
                d[cname] = metrics.paired_difference(ea[m], eb[m], clusters[m], n_boot)
            diffs[f"{names[i]} - {names[j]}"] = d
    out["paired_absrel_difference"] = diffs
    return out


def summarize(rows, n_boot=2000):
    """The distance tables from per-object rows (one split, or all folds pooled)."""
    by_range = {**rows, "z": rows["range"]}  # same objects, scored against the Euclidean distance (binned by it)
    return {"distance": breakdown({m: rows[f"pred_{m}"] for m in METHODS}, rows, n_boot),
            "range": breakdown({m: rows[f"range_{m}"] for m in METHODS}, by_range, n_boot),
            # error floor of the geometric methods: the same formulas on the labeled boxes of the same objects
            "distance_on_label_boxes": breakdown({m: rows[f"label_{m}"] for m in ("size", "ground")}, rows, n_boot),
            "ground_fallback_share": float(np.mean(rows["fallback"]))}


def run(root, det_dir, split=None, n_boot=2000, seed=0, mlp_epochs=150, save_model=None):
    info, rows = predict(root, det_dir, split, seed, mlp_epochs, save_model)
    return {**info, **summarize(rows, n_boot)}, rows


def compare(a, b, n_boot=2000):
    """Two detectors on the objects both of them found: per method, mean AbsRel difference (b - a)."""
    common, ia, ib = np.intersect1d(a["keys"], b["keys"], return_indices=True)
    z = a["z"][ia]
    clusters = np.array([f"{s}/{t}" for s, t in zip(a["seq"][ia], a["track"][ia])])
    out = {"n_common": int(len(common)), "tracks": int(len(np.unique(clusters)))}
    for m in METHODS:
        k = f"pred_{m}"
        ea, eb = metrics.abs_rel(a[k][ia], z), metrics.abs_rel(b[k][ib], z)
        out[m] = {"a": metrics.summarize(a[k][ia], z, clusters, n_boot).get("absrel_mean"),
                  "b": metrics.summarize(b[k][ib], z, clusters, n_boot).get("absrel_mean"),
                  "b - a": metrics.paired_difference(eb, ea, clusters, n_boot)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--det", nargs="+", required=True, help="detection directories")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--save-model", action="store_true", help="also write <out>/<name>.distance.pt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in args.det:
        name = Path(d).name
        res, _ = run(args.root, d, n_boot=args.boot, save_model=out / f"{name}.distance.pt" if args.save_model else None)
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(name, "mAP50-95 %.3f" % res["detection"]["mAP50-95"],
              {m: round(res["distance"][m]["all"]["absrel_mean"][0], 4) for m in METHODS})


if __name__ == "__main__":
    main()
