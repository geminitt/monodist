"""Matching detections to labeled objects, detection AP, and the matched pairs the distance methods use."""
from collections import defaultdict

import numpy as np

from . import kitti
from .geometry import iou

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
TP, FP, IGNORED = 1, 0, -1


def ground_truth(root, seqs):
    """Per (seq, frame): labeled objects of the class groups, ignore boxes (Misc) and DontCare boxes."""
    gt = {}
    for seq in seqs:
        lab = kitti.load_labels(root, seq)
        loc = kitti.to_camera2(lab["loc"], kitti.load_calib(root, seq))
        rng = kitti.center_range(lab["dim"], loc)
        cls = kitti.classify(lab["type"])
        for fr in range(kitti.FRAMES[seq]):
            m = lab["frame"] == fr
            obj = m & (cls >= 0)
            gt[(seq, fr)] = {
                "box": lab["box"][obj],
                "cls": cls[obj],
                "track": lab["track"][obj],
                "z": loc[obj, 2],
                "range": rng[obj],
                "trunc": lab["trunc"][obj],
                "occ": lab["occ"][obj],
                "ignore_box": lab["box"][m & np.isin(lab["type"], kitti.IGNORE)],
                "dontcare": lab["box"][m & (lab["type"] == "DontCare")],
            }
    return gt


def regroup(det, gt, mapping):
    """Copies with every class index c replaced by mapping[c] (e.g. kitti.GROUP_OF): matching then scores groups."""
    mapping = np.asarray(mapping, int)
    det = {**det, "cls": mapping[np.asarray(det["cls"], int)]}
    gt = {k: {**g, "cls": mapping[g["cls"]]} for k, g in gt.items()}
    return det, gt


def agnostic(det, gt):
    """Copies with every class set to 0: matching then asks only whether an object was found, not its class."""
    return regroup(det, gt, np.zeros(len(kitti.CLASSES), int))


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
    ign = iou(det_box, g["ignore_box"]).max(1) if len(g["ignore_box"]) else np.zeros(D)
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
            if ign[d] >= thr or dontcare[d]:
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


def detection_report(det, gt, status, thresholds=IOU_THRESHOLDS, names=kitti.CLASSES):
    """AP per class at each IoU threshold, AP50, and mAP50-95 averaged over the classes that have labels.

    A class without labeled objects in the evaluated sequences has AP NaN and is left out of the means;
    "classes" lists the ones averaged.
    """
    out = {}
    for c, name in enumerate(names):
        n_gt = int(sum((g["cls"] == c).sum() for g in gt.values()))
        m = det["cls"] == c
        aps = [average_precision(det["score"][m], status[m, t], n_gt) for t in range(len(thresholds))]
        out[name] = {"n_gt": n_gt, "AP50": aps[0], "AP50-95": float(np.mean(aps))}
    present = [n for n in names if out[n]["n_gt"] > 0]
    out["classes"] = present
    out["mAP50"] = float(np.mean([out[n]["AP50"] for n in present])) if present else float("nan")
    out["mAP50-95"] = float(np.mean([out[n]["AP50-95"] for n in present])) if present else float("nan")
    return out


def f1_threshold(score, status, n_gt):
    """The score threshold that maximizes F1 (IoU >= 0.5 status column), ignoring IGNORED detections."""
    keep = status != IGNORED
    score, tp = score[keep], status[keep] == TP
    if n_gt == 0 or len(score) == 0:
        return 1.0
    order = np.argsort(-score, kind="stable")
    score, ctp = score[order], np.cumsum(tp[order])
    f1 = 2 * ctp / (np.arange(1, len(score) + 1) + n_gt)
    # score >= t keeps every detection tied with t, so only the last of each run of equal scores is reachable
    last = np.append(score[1:] != score[:-1], True)
    return float(score[int(np.argmax(np.where(last, f1, -1)))])


def per_detection_conf(det, thresholds):
    """Broadcast one threshold per class to one threshold per detection."""
    return np.asarray(thresholds)[np.asarray(det["cls"], int)]


def matched_pairs(det, gt, status, gt_idx, conf, iou_col=0):
    """One row per detection that hits a labeled object at IoU >= 0.5 with score >= conf (scalar or per detection)."""
    rows = np.where((status[:, iou_col] == TP) & (det["score"] >= conf))[0]
    cols = defaultdict(list)
    for i in rows:
        g = gt[(det["seq"][i], det["frame"][i])]
        j = gt_idx[i, iou_col]
        for k in ("z", "range", "track", "trunc", "occ"):
            cols[k].append(g[k][j])
        cols["gt_box"].append(g["box"][j])
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("seq", "frame", "box", "cls", "score"):
        out[k] = det[k][rows]
    if len(rows) == 0:
        out.update(z=np.zeros(0), range=np.zeros(0), track=np.zeros(0, int), trunc=np.zeros(0, int), occ=np.zeros(0, int),
                   gt_box=np.zeros((0, 4)))
    return out


def recall_by_distance(det, gt, status, gt_idx, conf, bins, iou_col=0, names=kitti.CLASSES):
    """Share of labeled objects found (score >= conf, IoU >= 0.5), per class and distance bin (depth Z)."""
    hit = set()
    for i in np.where((status[:, iou_col] == TP) & (det["score"] >= conf))[0]:
        hit.add((det["seq"][i], det["frame"][i], gt_idx[i, iou_col]))
    out = {}
    for c, name in enumerate(names):
        found, total = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
        for key, g in gt.items():
            for j in np.where(g["cls"] == c)[0]:
                b = np.searchsorted(bins, g["z"][j], side="right") - 1
                if 0 <= b < len(bins) - 1:
                    total[b] += 1
                    found[b] += (key[0], key[1], j) in hit
        out[name] = {"found": found.tolist(), "total": total.tolist()}
    return out
