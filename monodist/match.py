"""Matching detections to labelled objects, detection AP, and the matched pairs the distance methods use."""
from collections import defaultdict

import numpy as np

from . import kitti
from .geometry import iou

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
TP, FP, IGNORED = 1, 0, -1


def ground_truth(root, seqs):
    """Per (seq, frame): labelled objects of the two classes, neighbour boxes and DontCare boxes."""
    gt = {}
    for seq in seqs:
        lab = kitti.load_labels(root, seq)
        P2 = kitti.load_calib(root, seq)
        z = kitti.to_camera2(lab["loc"], P2)[:, 2]
        for fr in range(kitti.FRAMES[seq]):
            m = lab["frame"] == fr
            t = lab["type"][m]
            obj = np.isin(t, kitti.CLASSES)
            gt[(seq, fr)] = {
                "box": lab["box"][m][obj],
                "cls": np.array([kitti.CLASSES.index(c) for c in t[obj]], int),
                "track": lab["track"][m][obj],
                "z": z[m][obj],
                "trunc": lab["trunc"][m][obj],
                "occ": lab["occ"][m][obj],
                "neighbour_box": lab["box"][m][np.isin(t, list(kitti.NEIGHBOURS))],
                "neighbour_cls": np.array([kitti.CLASSES.index(kitti.NEIGHBOURS[c]) for c in t if c in kitti.NEIGHBOURS], int),
                "dontcare": lab["box"][m][t == "DontCare"],
            }
    return gt


def _covered(det_box, regions):
    """Fraction of each detection's area inside its most-covering region (the devkit's DontCare criterion)."""
    if len(regions) == 0 or len(det_box) == 0:
        return np.zeros(len(det_box))
    d, r = det_box[:, None, :], regions[None, :, :]
    ix = np.clip(np.minimum(d[..., 2], r[..., 2]) - np.maximum(d[..., 0], r[..., 0]), 0, None)
    iy = np.clip(np.minimum(d[..., 3], r[..., 3]) - np.maximum(d[..., 1], r[..., 1]), 0, None)
    area = (det_box[:, 2] - det_box[:, 0]) * (det_box[:, 3] - det_box[:, 1])
    return (ix * iy / np.maximum(area[:, None], 1e-9)).max(1)


def match_frame(det_box, det_score, det_cls, g, thresholds=IOU_THRESHOLDS):
    """Greedy matching in score order, per class and IoU threshold.

    Returns status (D, T) in {TP, FP, IGNORED} and matched ground-truth index (D, T), -1 when unmatched.
    """
    D, T = len(det_box), len(thresholds)
    status = np.full((D, T), FP, int)
    gt_idx = np.full((D, T), -1, int)
    if D == 0:
        return status, gt_idx
    order = np.argsort(-det_score, kind="stable")
    ious = iou(det_box, g["box"])
    nious = iou(det_box, g["neighbour_box"])
    dontcare = _covered(det_box, g["dontcare"]) >= 0.5
    for ti, thr in enumerate(thresholds):
        taken = np.zeros(len(g["box"]), bool)
        for d in order:
            cand = (g["cls"] == det_cls[d]) & ~taken
            if cand.any():
                j = np.where(cand, ious[d], -1).argmax()
                if ious[d, j] >= thr:
                    status[d, ti], gt_idx[d, ti] = TP, j
                    taken[j] = True
                    continue
            same = g["neighbour_cls"] == det_cls[d]
            if (same.any() and nious[d, same].max() >= thr) or dontcare[d]:
                status[d, ti] = IGNORED
    return status, gt_idx


def evaluate(det, gt, thresholds=IOU_THRESHOLDS):
    """Match every frame. det: dict of arrays seq, frame, box, score, cls (one row per detection)."""
    by_frame = defaultdict(list)
    for i, key in enumerate(zip(det["seq"], det["frame"])):
        by_frame[key].append(i)
    status = np.full((len(det["score"]), len(thresholds)), FP, int)
    gt_idx = np.full_like(status, -1)
    for key, g in gt.items():
        idx = np.array(by_frame.get(key, []), int)
        if len(idx):
            status[idx], gt_idx[idx] = match_frame(det["box"][idx], det["score"][idx], det["cls"][idx], g, thresholds)
    extra = set(by_frame) - set(gt)
    assert not extra, f"detections on frames without ground truth, e.g. {next(iter(extra))}"
    return status, gt_idx


def average_precision(score, status, n_gt):
    """COCO-style AP: 101-point interpolated area under the precision-recall curve."""
    keep = status != IGNORED
    score, tp = score[keep], (status[keep] == TP)
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        i = np.searchsorted(recall, r, side="left")
        ap += envelope[i] if i < len(envelope) else 0.0
    return ap / 101


def detection_report(det, gt, status, thresholds=IOU_THRESHOLDS):
    """AP per class at each IoU threshold, AP50, and mAP50-95 averaged over classes and thresholds."""
    out = {}
    for c, name in enumerate(kitti.CLASSES):
        n_gt = int(sum((g["cls"] == c).sum() for g in gt.values()))
        m = det["cls"] == c
        aps = [average_precision(det["score"][m], status[m, t], n_gt) for t in range(len(thresholds))]
        out[name] = {"n_gt": n_gt, "AP50": aps[0], "AP50-95": float(np.mean(aps))}
    out["mAP50"] = float(np.mean([out[n]["AP50"] for n in kitti.CLASSES]))
    out["mAP50-95"] = float(np.mean([out[n]["AP50-95"] for n in kitti.CLASSES]))
    return out


def matched_pairs(det, gt, status, gt_idx, conf, iou_col=0):
    """One row per detection that hits a labelled object at IoU >= 0.5 with score >= conf."""
    rows = np.where((status[:, iou_col] == TP) & (det["score"] >= conf))[0]
    cols = defaultdict(list)
    for i in rows:
        g = gt[(det["seq"][i], det["frame"][i])]
        j = gt_idx[i, iou_col]
        for k in ("z", "track", "trunc", "occ"):
            cols[k].append(g[k][j])
        cols["gt_box"].append(g["box"][j])
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("seq", "frame", "box", "cls", "score"):
        out[k] = det[k][rows]
    if len(rows) == 0:
        out["gt_box"] = np.zeros((0, 4))
    return out


def recall_by_distance(det, gt, status, gt_idx, conf, bins, iou_col=0):
    """Share of labelled objects found (score >= conf, IoU >= 0.5), per class and distance bin."""
    hit = set()
    for i in np.where((status[:, iou_col] == TP) & (det["score"] >= conf))[0]:
        hit.add((det["seq"][i], det["frame"][i], gt_idx[i, iou_col]))
    out = {}
    for c, name in enumerate(kitti.CLASSES):
        found, total = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
        for key, g in gt.items():
            for j in np.where(g["cls"] == c)[0]:
                b = np.searchsorted(bins, g["z"][j], side="right") - 1
                if 0 <= b < len(bins) - 1:
                    total[b] += 1
                    found[b] += (key[0], key[1], j) in hit
        out[name] = {"found": found.tolist(), "total": total.tolist()}
    return out
