"""Evaluate detectors and the three distance methods on the test sequences; write one JSON per run.

    python -m monodist.evaluate --root data/kitti_tracking --det results/det/coco_1280 --det results/det/ft_1280

Everything here runs on CPU from saved detections: fitting the geometric constants on the train labels,
training the distance MLP on this detector's boxes on the train sequences, and scoring on the test sequences.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import geometry, kitti, match, metrics, mlp
from .detect import merge

CONF = 0.25  # detections used for distance: the usual operating point of a YOLO detector
MIN_Z = 2.0  # closer objects are all cut by the image border; AbsRel near Z = 0 would dominate the mean


def constants(root):
    """Known-size heights per class and the camera height above the road, from the train labels only."""
    heights = {c: [] for c in kitti.CLASSES}
    road = []
    for seq in kitti.SPLIT["train"]:
        lab = kitti.load_labels(root, seq)
        y = kitti.to_camera2(lab["loc"], kitti.load_calib(root, seq))[:, 1]
        for c in kitti.CLASSES:
            heights[c].extend(lab["dim"][lab["type"] == c, 0])
        road.extend(y[lab["type"] == "Car"])
    return {"class_height": [float(np.mean(heights[c])) for c in kitti.CLASSES], "cam_height": float(np.median(road))}


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


def feats(pairs, cam):
    s = pairs["seq"]
    return geometry.features(pairs["box"], pairs["cls"], per_row(cam, s, 0), per_row(cam, s, 1), per_row(cam, s, 2),
                             per_row(cam, s, 3), per_row(cam, s, 4))


def breakdown(preds, pairs, n_boot):
    """Every method, overall and per class x distance bin, plus paired differences between methods."""
    z, clusters = pairs["z"], np.array([f"{s}/{t}" for s, t in zip(pairs["seq"], pairs["track"])])
    b = metrics.distance_bin(z)
    out = {}
    for name, p in preds.items():
        out[name] = {"all": metrics.summarize(p, z, clusters, n_boot)}
        for c, cname in enumerate(kitti.CLASSES):
            m = pairs["cls"] == c
            out[name][cname] = metrics.summarize(p[m], z[m], clusters[m], n_boot)
            whole = m & (pairs["trunc"] == 0)  # not cut by the image border (KITTI truncation level 0)
            out[name][f"{cname} not truncated"] = metrics.summarize(p[whole], z[whole], clusters[whole], n_boot)
            for k, bname in enumerate(metrics.BIN_NAMES):
                mb = m & (b == k)
                out[name][f"{cname} {bname}"] = metrics.summarize(p[mb], z[mb], clusters[mb], n_boot)
    diffs = {}
    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ea, eb = metrics.abs_rel(preds[names[i]], z), metrics.abs_rel(preds[names[j]], z)
            diffs[f"{names[i]} - {names[j]}"] = {"all": metrics.paired_difference(ea, eb, clusters, n_boot)}
            for c, cname in enumerate(kitti.CLASSES):
                m = pairs["cls"] == c
                diffs[f"{names[i]} - {names[j]}"][cname] = metrics.paired_difference(ea[m], eb[m], clusters[m], n_boot)
    out["paired_absrel_difference"] = diffs
    return out


def subset(d, seqs):
    m = np.isin(d["seq"], seqs)
    return {k: v[m] for k, v in d.items()}


def run(root, det_dir, const, n_boot=2000, seed=0, mlp_epochs=150):
    det, sizes = merge(det_dir)
    seqs = kitti.SEQUENCES
    missing = sorted(set(seqs) - set(sizes))
    assert not missing, f"{det_dir}: no detections for sequences {missing} (all 21 are needed)"
    det = subset(det, seqs)
    cam = camera(root, seqs, sizes)
    gt = match.ground_truth(root, seqs)
    status, gt_idx = match.evaluate(det, gt)
    test_gt = {k: v for k, v in gt.items() if k[0] in kitti.SPLIT["test"]}
    tm = np.isin(det["seq"], kitti.SPLIT["test"])
    tdet = {k: v[tm] for k, v in det.items()}
    res = {"det_dir": str(det_dir), "conf": CONF, "min_z": MIN_Z, "constants": const,
           "detection": match.detection_report(tdet, test_gt, status[tm]),
           "recall_by_distance": match.recall_by_distance(tdet, test_gt, status[tm], gt_idx[tm], CONF, metrics.BINS)}

    pairs = match.matched_pairs(det, gt, status, gt_idx, CONF)
    near = pairs["z"] < MIN_Z
    res["excluded_below_min_z"] = int(near.sum())
    pairs = {k: v[~near] for k, v in pairs.items()}
    train, test = subset(pairs, kitti.SPLIT["train"]), subset(pairs, kitti.SPLIT["test"])
    # How tight this detector's boxes are on each split: a fine-tuned detector is tighter on images it trained on,
    # which would make an MLP trained on those boxes trust the box height too much.
    res["box_iou_median"] = {sp: float(np.median(geometry.iou_pairs(p["box"], p["gt_box"])))
                             for sp, p in ((sp, subset(pairs, kitti.SPLIT[sp])) for sp in kitti.SPLIT)}
    model, epochs, curve = mlp.train(feats(train, cam), train["z"], train["seq"], max_epochs=mlp_epochs, seed=seed)
    res["mlp"] = {"epochs": epochs, "heldout_l1_log": [float(v) for v in curve], "n_train": int(len(train["z"]))}

    za, zb, fallback = geometric(test, test["box"], cam, const)
    preds = {"size": za, "ground": zb, "mlp": mlp.predict(model, feats(test, cam))}
    res["ground_fallback_share"] = float(fallback.mean())
    res["distance"] = breakdown(preds, test, n_boot)
    # Error floor of the geometric methods: the same formulas on the labelled boxes of the same objects.
    ga, gb, _ = geometric(test, test["gt_box"], cam, const)
    res["distance_on_label_boxes"] = breakdown({"size": ga, "ground": gb}, test, n_boot)
    keys = [f"{s}/{f}/{t}" for s, f, t in zip(test["seq"], test["frame"], test["track"])]
    return res, {"keys": np.array(keys), "z": test["z"], "cls": test["cls"], "seq": test["seq"],
                 "track": test["track"], **{f"pred_{k}": v for k, v in preds.items()}}


def compare(a, b, n_boot=2000):
    """Two detectors on the objects both of them found: per method, mean AbsRel difference (b - a)."""
    common, ia, ib = np.intersect1d(a["keys"], b["keys"], return_indices=True)
    z = a["z"][ia]
    clusters = np.array([f"{s}/{t}" for s, t in zip(a["seq"][ia], a["track"][ia])])
    out = {"n_common": int(len(common)), "tracks": int(len(np.unique(clusters)))}
    for k in [k for k in a if k.startswith("pred_")]:
        ea, eb = metrics.abs_rel(a[k][ia], z), metrics.abs_rel(b[k][ib], z)
        out[k[5:]] = {"a": metrics.summarize(a[k][ia], z, clusters, n_boot)["absrel_mean"],
                      "b": metrics.summarize(b[k][ib], z, clusters, n_boot)["absrel_mean"],
                      "b - a": metrics.paired_difference(eb, ea, clusters, n_boot)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--det", nargs="+", required=True, help="detection directories; the first is the baseline")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    const = constants(args.root)
    rows = {}
    for d in args.det:
        res, rows[d] = run(args.root, d, const, args.boot)
        name = Path(d).name
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(name, "mAP50-95 %.3f" % res["detection"]["mAP50-95"],
              {k: round(v["all"]["absrel_mean"][0], 4) for k, v in res["distance"].items() if "all" in v and k != "paired_absrel_difference"})
    base = args.det[0]
    for d in args.det[1:]:
        c = compare(rows[base], rows[d], args.boot)
        (out / f"compare_{Path(base).name}_vs_{Path(d).name}.json").write_text(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
