"""Three-fold cross-validation by sequence (kitti.FOLDS).

    python -m monodist.crossval --root data/kitti_tracking --runs results/cv/det --configs a b c d --out results/cv

For each fold, the fine-tuning configuration is chosen by mAP50-95 on that fold's val sequences (with the same
matcher and ignore rules as the test metric, and the NMS head); the chosen detector is then evaluated on the
fold's test sequences. Every sequence is test exactly once, so the per-object distance rows of the three folds
pool into one set covering all 21 sequences. Detection AP is computed per fold (scores of different models
are not comparable) and reported with its mean and spread over folds.

Detection directories are named <runs>/f<fold>_<config>_<head>, head "nms" or "e2e"; the COCO baseline uses
the same detections in every fold (it is never trained).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import evaluate, kitti, match
from .detect import merge


def val_map(root, det_dir, fold):
    """mAP50-95 on the fold's val sequences, KITTI ignore rules included."""
    det, _ = merge(det_dir)
    val = kitti.FOLDS[fold]["val"]
    det = evaluate.subset(det, val)
    gt = match.ground_truth(root, val)
    status, _ = match.evaluate(det, gt)
    return match.detection_report(det, gt, status)["mAP50-95"]


def select(root, runs, configs, head="nms"):
    table = {}
    for k in range(len(kitti.FOLDS)):
        table[k] = {c: val_map(root, Path(runs) / f"f{k}_{c}_{head}", k) for c in configs
                    if (Path(runs) / f"f{k}_{c}_{head}").is_dir()}
    best = {k: max(t, key=t.get) for k, t in table.items()}
    return table, best


def test_map(root, det_dir, fold):
    det, _ = merge(det_dir)
    test = kitti.FOLDS[fold]["test"]
    det = evaluate.subset(det, test)
    gt = match.ground_truth(root, test)
    status, _ = match.evaluate(det, gt)
    return match.detection_report(det, gt, status)["mAP50-95"]


def best_epoch(results_csv):
    """The epoch Ultralytics kept: highest fitness = 0.1 mAP50 + 0.9 mAP50-95 on val."""
    import csv
    rows = list(csv.DictReader(open(results_csv)))
    return int(max(rows, key=lambda r: 0.1 * float(r["metrics/mAP50(B)"]) + 0.9 * float(r["metrics/mAP50-95(B)"]))["epoch"])


def hypothesis(root, runs, logs, fold=0, configs=("a", "e")):
    """Does choosing the epoch with the NMS-free head (e) fix that head, compared with choosing it with NMS (a)?"""
    out = {}
    for c in configs:
        d = {f"{part}_{head}": fn(root, Path(runs) / f"f{fold}_{c}_{head}", fold)
             for part, fn in (("val", val_map), ("test", test_map)) for head in ("nms", "e2e")}
        d["epoch"] = best_epoch(Path(logs) / f"f{fold}-{c}" / "runs" / f"f{fold}_{c}" / "results.csv")
        out[c] = d
    return out


def pool(parts):
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def cross_validate(root, det_dirs, n_boot=2000, mlp_epochs=150):
    """det_dirs: one detection directory per fold. Returns the pooled report and the pooled per-object rows."""
    infos, rows = [], []
    for k, d in enumerate(det_dirs):
        info, r = evaluate.predict(root, d, kitti.FOLDS[k], mlp_epochs=mlp_epochs)
        infos.append(info)
        rows.append(r)
    rows = pool(rows)
    assert len(np.unique(rows["keys"])) == len(rows["keys"]), "an object appears in two test folds"
    per_fold = {m: [i["detection"][m] for i in infos] for m in ("mAP50-95", "mAP50")}
    for c in kitti.CLASSES:
        per_fold[f"{c} AP50-95"] = [i["detection"][c]["AP50-95"] for i in infos]
    detection = {m: {"per_fold": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for m, v in per_fold.items()}
    recall = {c: {part: np.sum([i["recall_by_distance"][c][part] for i in infos], axis=0).tolist()
                  for part in ("found", "total")} for c in kitti.CLASSES}
    report = {"det_dirs": [str(d) for d in det_dirs], "detection": detection, "recall_by_distance": recall,
              "folds": [{k: i[k] for k in ("conf", "constants", "mlp", "box_iou_median", "excluded_below_min_z",
                                           "val_detection")} for i in infos],
              **evaluate.summarize(rows, n_boot)}
    return report, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", required=True, help="directory holding f<fold>_<config>_<head> detection folders")
    ap.add_argument("--coco", default="results/det/coco_1280", help="COCO-weights detections (all 21 sequences)")
    ap.add_argument("--configs", nargs="+", default=["a", "b", "c", "d"])
    ap.add_argument("--out", default="results/cv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--logs", default="runs/kaggle/cv", help="downloaded Kaggle outputs (training curves)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table, best = select(args.root, args.runs, args.configs)
    (out / "selection.json").write_text(json.dumps({"val_mAP50-95": table, "chosen": best}, indent=1))
    print("val mAP50-95 per fold and configuration:", table, "-> chosen", best)

    runs = Path(args.runs)
    if (runs / "f0_e_nms").is_dir():
        h = hypothesis(args.root, runs, args.logs)
        (out / "hypothesis_e.json").write_text(json.dumps(h, indent=1))
        print("hypothesis e vs a (fold 0):", h)
    detectors = {"coco": [args.coco] * len(kitti.FOLDS),
                 "finetuned_nms": [runs / f"f{k}_{best[k]}_nms" for k in range(len(kitti.FOLDS))],
                 "finetuned_e2e": [runs / f"f{k}_{best[k]}_e2e" for k in range(len(kitti.FOLDS))]}
    rows = {}
    for name, dirs in detectors.items():
        report, rows[name] = cross_validate(args.root, dirs, args.boot)
        (out / f"{name}.json").write_text(json.dumps(report, indent=1))
        print(name, "mAP50-95 %.3f +- %.3f" % (report["detection"]["mAP50-95"]["mean"], report["detection"]["mAP50-95"]["std"]),
              {m: round(report["distance"][m]["all"]["absrel_mean"][0], 4) for m in evaluate.METHODS})
    for name in ("finetuned_nms", "finetuned_e2e"):
        (out / f"compare_coco_vs_{name}.json").write_text(json.dumps(evaluate.compare(rows["coco"], rows[name], args.boot), indent=1))


if __name__ == "__main__":
    main()
