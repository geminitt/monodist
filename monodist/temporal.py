"""Time: a tracker, a Kalman filter on each object's distance, and time to collision (TTC).

    python -m monodist.temporal --root data/kitti_tracking --rows results/cv/finetuned_nms_rows.npz

A single frame gives a distance with a relative error of several percent. The closing speed is the change of
distance over time, and differencing two noisy distances 0.1 s apart amplifies that error. The filter keeps,
per object, a state (Z, dZ/dt) under a constant-velocity model and blends each new measurement into it,
weighting it by how noisy measurements are (relative error meas_rel) against how fast the speed may change
(acceleration accel). TTC = Z / (-dZ/dt) for an approaching object (dZ/dt < 0), infinite otherwise.

The filter is causal (it uses only past frames), as it must be in a running system. Its two settings are
chosen per fold on the other folds' sequences, so every reported number is on sequences the choice never saw.
"""
import argparse
import itertools
import json
from collections import defaultdict

import numpy as np

from . import geometry, kitti

DT = 0.1  # KITTI tracking: 10 frames per second
# Grid of the filter settings chosen on the development folds. On KITTI label boxes the choice lands at the
# corner "trust each distance, keep the speed nearly constant": the MLP's error is mostly a bias that persists
# from frame to frame of one object, so it cancels in the speed, and a long effective window averages the rest.
MEAS_REL = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
ACCEL = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)  # m/s^2
TTC_MAX = 10.0  # s; beyond it a TTC estimate has no use for braking decisions
WINDOW = 5  # frames of the least-squares slope that gives the labeled speed


class DistanceFilter:
    """Constant-velocity Kalman filter on (Z, dZ/dt); measurement noise proportional to Z."""

    def __init__(self, z, meas_rel, accel, v0_sd=5.0):
        self.x = np.array([z, 0.0])
        self.P = np.diag([(meas_rel * z) ** 2, v0_sd ** 2])
        self.meas_rel, self.accel = meas_rel, accel

    def predict(self, dt):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([dt ** 2 / 2, dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.outer(G, G) * self.accel ** 2

    def update(self, z):
        r = (self.meas_rel * max(z, 1e-3)) ** 2
        s = self.P[0, 0] + r
        k = self.P[:, 0] / s
        self.x = self.x + k * (z - self.x[0])
        self.P = self.P - np.outer(k, self.P[0])


def filter_track(frames, z, meas_rel, accel):
    """Filtered Z and dZ/dt after each measurement of one track (frames ascending; gaps allowed)."""
    kf = DistanceFilter(z[0], meas_rel, accel)
    zf, vf = [kf.x[0]], [kf.x[1]]
    for k in range(1, len(z)):
        kf.predict((frames[k] - frames[k - 1]) * DT)
        kf.update(z[k])
        zf.append(kf.x[0])
        vf.append(kf.x[1])
    return np.array(zf), np.array(vf)


def ttc(z, v):
    """Time to collision; infinite for objects that are not approaching."""
    z, v = np.asarray(z, float), np.asarray(v, float)
    return np.where(v < -1e-6, z / np.maximum(-v, 1e-6), np.inf)


class Tracker:
    """Greedy IoU association frame to frame; a track ends after max_age frames without a detection."""

    def __init__(self, meas_rel, accel, iou_min=0.3, max_age=3):
        self.meas_rel, self.accel, self.iou_min, self.max_age = meas_rel, accel, iou_min, max_age
        self.tracks = {}  # id -> dict(box, cls, filter, last)
        self.next_id = 0

    def update(self, frame, box, cls, z):
        """Associate this frame's objects; returns track id, filtered Z and dZ/dt per object."""
        ids, zf, vf = np.full(len(box), -1), np.zeros(len(box)), np.zeros(len(box))
        live = [t for t in self.tracks if frame - self.tracks[t]["last"] <= self.max_age]
        if live and len(box):
            iou = geometry.iou(box, np.array([self.tracks[t]["box"] for t in live]))
            same = cls[:, None] == np.array([self.tracks[t]["cls"] for t in live])[None, :]
            iou = np.where(same, iou, 0)
            for i, j in sorted(zip(*np.nonzero(iou >= self.iou_min)), key=lambda p: -iou[p]):
                if ids[i] < 0 and live[j] not in ids:
                    ids[i] = live[j]
        for i in range(len(box)):
            if ids[i] < 0:
                ids[i] = self.next_id
                self.tracks[self.next_id] = {"filter": DistanceFilter(z[i], self.meas_rel, self.accel), "last": frame}
                self.next_id += 1
            else:
                t = self.tracks[ids[i]]
                t["filter"].predict((frame - t["last"]) * DT)
                t["filter"].update(z[i])
            t = self.tracks[ids[i]]
            t.update(box=box[i], cls=cls[i], last=frame)
            zf[i], vf[i] = t["filter"].x
        self.tracks = {k: t for k, t in self.tracks.items() if frame - t["last"] <= self.max_age}
        return ids, zf, vf


def labeled_speed(root, seqs):
    """dZ/dt of every labeled object per (seq, track, frame): least-squares slope over WINDOW frames."""
    out = {}
    half = WINDOW // 2
    for seq in seqs:
        lab = kitti.load_labels(root, seq)
        z = kitti.to_camera2(lab["loc"], kitti.load_calib(root, seq))[:, 2]
        for t in np.unique(lab["track"]):
            m = lab["track"] == t
            fr, zt = lab["frame"][m], z[m]
            for k in range(len(fr)):
                w = np.abs(fr - fr[k]) <= half
                if w.sum() >= 3:
                    out[(seq, int(t), int(fr[k]))] = float(np.polyfit(fr[w] * DT, zt[w], 1)[0])
    return out


def run_rows(rows, meas_rel, accel):
    """Filter every track of per-object rows (seq, track, frame, pred_mlp); returns filtered Z and dZ/dt."""
    zf, vf = np.full(len(rows["pred_mlp"]), np.nan), np.full(len(rows["pred_mlp"]), np.nan)
    groups = defaultdict(list)
    for i, key in enumerate(zip(rows["seq"], rows["track"])):
        groups[key].append(i)
    for idx in groups.values():
        idx = np.array(idx)[np.argsort(rows["frame"][idx], kind="stable")]
        zf[idx], vf[idx] = filter_track(rows["frame"][idx], rows["pred_mlp"][idx], meas_rel, accel)
    return zf, vf


def raw_speed(rows):
    """dZ/dt from consecutive raw predictions of the same track (the unfiltered baseline); NaN on a track's first row."""
    v = np.full(len(rows["pred_mlp"]), np.nan)
    groups = defaultdict(list)
    for i, key in enumerate(zip(rows["seq"], rows["track"])):
        groups[key].append(i)
    for idx in groups.values():
        idx = np.array(idx)[np.argsort(rows["frame"][idx], kind="stable")]
        fr, z = rows["frame"][idx], rows["pred_mlp"][idx]
        v[idx[1:]] = np.diff(z) / (np.diff(fr) * DT)
    return v


def scores(rows, zf, vf, v_true, v_raw):
    """Raw against filtered on the same rows (those with a labeled speed and a previous raw prediction):
    distance AbsRel, speed MAE, and TTC error on approaching objects with a labeled TTC within TTC_MAX."""
    ok = ~np.isnan(v_true) & ~np.isnan(vf) & ~np.isnan(v_raw)
    z = rows["z"]
    t_true = ttc(z, v_true)
    near = ok & (t_true < TTC_MAX)
    out = {"n": int(ok.sum()), "ttc_n": int(near.sum())}
    for name, zz, vv in (("raw", rows["pred_mlp"], v_raw), ("filtered", zf, vf)):
        rel = np.abs(np.minimum(ttc(zz, vv)[near], 1e3) - t_true[near]) / t_true[near]
        out[name] = {"absrel": float(np.mean(np.abs(zz[ok] - z[ok]) / z[ok])),
                     "speed_mae": float(np.mean(np.abs(vv[ok] - v_true[ok]))),
                     "ttc_median_rel_error": float(np.median(rel)) if near.any() else None,
                     "ttc_within_20pct": float(np.mean(rel <= 0.2)) if near.any() else None}
    return out


def associate(rows):
    """Run the actual tracker on every kept box, including false positives and ignored regions.

    Association depends only on sequence, frame, box and predicted class. KITTI track ids are scoring labels.
    Returns tracker ids (local to a sequence); filter settings do not affect IoU association.
    """
    ids = np.full(len(rows["frame"]), -1, int)
    for seq in np.unique(rows["seq"]):
        tracker = Tracker(meas_rel=0.01, accel=1.0)
        indices = np.flatnonzero(rows["seq"] == seq)
        indices = indices[np.argsort(rows["frame"][indices], kind="stable")]
        frames, starts = np.unique(rows["frame"][indices], return_index=True)
        for frame, idx in zip(frames, np.split(indices, starts[1:])):
            ids[idx], _, _ = tracker.update(int(frame), rows["box"][idx], rows["cls"][idx], rows["pred_mlp"][idx])
    return ids


def score_observations(rows, ids, speed, meas_rel, accel):
    """Score automatic tracks; speed and KITTI ids are consulted only after filtering."""
    predicted = {**rows, "track": ids}
    zf, vf = run_rows(predicted, meas_rel, accel)
    vr = raw_speed(predicted)
    vt = np.array([speed.get((s, int(t), int(f)), np.nan)
                   for s, t, f in zip(rows["seq"], rows["track"], rows["frame"])])
    valid = (rows["track"] >= 0) & (rows["z"] >= 2.0)
    vt[~valid] = np.nan
    result = scores(rows, zf, vf, vt, vr)
    result["detections"] = len(ids)
    result["matched_at_least_2m"] = int(valid.sum())
    result["with_history_fraction"] = float(np.mean(np.isfinite(vr[valid]))) if valid.any() else None
    result["distance_all_matched"] = {
        name: float(np.mean(np.abs(z[valid] - rows["z"][valid]) / rows["z"][valid])) if valid.any() else None
        for name, z in (("raw", rows["pred_mlp"]), ("filtered", zf))}
    return result, zf, vf


def association_summary(rows, ids):
    """Descriptive identity continuity, not an official MOT benchmark (no HOTA/MOTA claim)."""
    valid = rows["track"] >= 0
    gt_groups, predicted_groups = defaultdict(list), defaultdict(set)
    for i in np.flatnonzero(valid):
        gt_groups[(rows["seq"][i], rows["track"][i])].append(i)
        predicted_groups[(rows["seq"][i], ids[i])].add(int(rows["track"][i]))
    changes, transitions, split = 0, 0, 0
    for idx in gt_groups.values():
        idx = np.array(idx)[np.argsort(rows["frame"][idx], kind="stable")]
        close = np.diff(rows["frame"][idx]) <= 3  # same gap tolerance as Tracker.max_age
        changes += int(((np.diff(ids[idx]) != 0) & close).sum())
        transitions += int(close.sum())
        split += len(np.unique(ids[idx])) > 1
    return {"matched_ground_truth_tracks": len(gt_groups), "nearby_matched_transitions": transitions,
            "identity_changes_across_nearby_matches": changes, "ground_truth_tracks_with_multiple_ids": split,
            "predicted_tracks_matching_multiple_objects": sum(len(v) > 1 for v in predicted_groups.values()),
            "unmatched_detections": int((rows["status"] == 0).sum()),
            "ignored_detections": int((rows["status"] == -1).sum())}


def select_filter(rows, ids, speed):
    """Choose the filter on this model's validation sequences, never on a fold's test observations."""
    table = []
    for meas_rel, accel in itertools.product(MEAS_REL, ACCEL):
        result, _, _ = score_observations(rows, ids, speed, meas_rel, accel)
        value = result["filtered"]["speed_mae"]
        if not np.isfinite(value):
            raise ValueError("no finite validation speed score; cannot select the filter")
        table.append({"meas_rel": meas_rel, "accel": accel, "speed_mae": value})
    return min(table, key=lambda r: r["speed_mae"]), table


def cross_validated(root, rows):
    """Per fold: choose (meas_rel, accel) by speed MAE on the other folds' sequences, score on this fold's."""
    speed = labeled_speed(root, sorted(set(rows["seq"])))
    v_true = np.array([speed.get((s, int(t), int(f)), np.nan) for s, t, f in zip(rows["seq"], rows["track"], rows["frame"])])
    vr = raw_speed(rows)
    out = {"folds": []}
    runs = {p: run_rows(rows, *p) for p in itertools.product(MEAS_REL, ACCEL)}
    zf_all, vf_all = np.full(len(rows["z"]), np.nan), np.full(len(rows["z"]), np.nan)
    for k, fold in enumerate(kitti.FOLDS):
        mine = np.isin(rows["seq"], fold["test"])
        other = ~mine & ~np.isnan(v_true)

        def mae(p):
            return np.nanmean(np.abs(runs[p][1][other] - v_true[other]))
        best = min(runs, key=mae)
        zf_all[mine], vf_all[mine] = runs[best][0][mine], runs[best][1][mine]
        out["folds"].append({"meas_rel": best[0], "accel": best[1], "other_folds_speed_mae": float(mae(best))})
    out.update(scores(rows, zf_all, vf_all, v_true, vr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rows", required=True, help="pooled cross-validation rows (.npz) of one detector")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = dict(np.load(args.rows))
    res = cross_validated(args.root, rows)
    print(json.dumps(res, indent=1))
    if args.out:
        open(args.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
