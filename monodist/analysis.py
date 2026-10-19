"""Resumable development-only distance diagnostics and evaluation of automatic tracks.

Each fold fits its own distance ensemble. Detector settings and confidence come from validation as before;
the Kalman settings are selected on that same model's validation predictions. Test labels score outputs only.
Held-out KITTI.TEST labels, images and detection files are never loaded. Run artifacts are signed by source,
environment, labels, calibration and detection hashes; a completed fold can be reused only with those inputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from . import crossval, evaluate, geometry, kitti, match, metrics, temporal
from .detect import merge


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    """Publish a complete JSON atomically, including a durable checkpoint before continuing."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def clean_json(value):
    """Missing classes and empty subsets are null, never nonstandard NaN tokens."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def identity(root, directories, epochs, boot):
    repo = Path(__file__).resolve().parents[1]
    data = kitti.training_dir(root)
    paths = sorted((repo / "monodist").glob("*.py")) + sorted((repo / "tests").glob("*.py"))
    paths += [repo / "pixi.lock", repo / "pixi.toml", repo / "pyproject.toml"]
    for seq in kitti.DEVELOPMENT:
        paths += [data / "label_02" / f"{seq}.txt", data / "calib" / f"{seq}.txt"]
    for directory in sorted(set(map(str, directories))):
        paths += [Path(directory) / "run.json"]
        paths += [Path(directory) / f"{seq}.npz" for seq in kitti.DEVELOPMENT]
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "files": {str(p): sha(p) for p in paths}, "folds": kitti.FOLDS,
            "epochs": epochs, "bootstrap_samples": boot,
            "python": sys.version,
            "environment": {p: importlib.metadata.version(p) for p in ("numpy", "torch", "ultralytics")}}


def diagnostics(rows):
    """Matched class-correct detections: bias, box size error, truncation and label-box counterfactual."""
    parts = {"all": np.ones(len(rows["z"]), bool)}
    for cls, name in enumerate(kitti.CLASSES):
        parts[name] = rows["cls"] == cls
        for b, bname in enumerate(metrics.BIN_NAMES):
            parts[f"{name} {bname}"] = parts[name] & (metrics.distance_bin(rows["z"]) == b)
        parts[f"{name} not truncated"] = parts[name] & (rows["trunc"] == 0)
        parts[f"{name} truncated"] = parts[name] & (rows["trunc"] > 0)
    result = {}
    for name, m in parts.items():
        z, p = rows["z"][m], rows["pred_mlp"][m]
        box, gt = rows["box"][m], rows["gt_box"][m]
        result[name] = {"n": int(m.sum())}
        if not m.any():
            continue
        result[name].update(
            tracks=len(set(zip(rows["seq"][m], rows["track"][m]))),
            absrel=float(np.mean(np.abs(p - z) / z)), signed_relative_bias=float(np.mean((p - z) / z)),
            label_box_mlp_absrel=float(np.mean(np.abs(rows["label_mlp"][m] - z) / z)),
            median_box_iou=float(np.median(geometry.iou_pairs(box, gt))),
            median_height_relative_error=float(np.median((box[:, 3] - box[:, 1]) / (gt[:, 3] - gt[:, 1]) - 1)))
    return result


def threshold_analysis(root, directory, fold, info):
    """Descriptive confidence tradeoffs on validation only; no new operating point is selected here."""
    det, _ = merge(directory, seqs=fold["val"])
    gt = match.ground_truth(root, fold["val"])
    status, _ = match.evaluate(det, gt, thresholds=[0.5])
    result = {}
    for c, name in enumerate(kitti.CLASSES):
        n_gt = sum(int((g["cls"] == c).sum()) for g in gt.values())
        points = {}
        for label, threshold in (("validation_f1", info["conf"][name]), ("fixed_0.1", 0.1),
                                 ("fixed_0.25", 0.25), ("fixed_0.5", 0.5)):
            m = (det["cls"] == c) & (det["score"] >= threshold) & (status[:, 0] != match.IGNORED)
            tp, n = int((status[m, 0] == match.TP).sum()), int(m.sum())
            points[label] = {"threshold": threshold, "tp": tp, "fp": n - tp, "fn": n_gt - tp,
                             "precision": tp / n if n else None, "recall": tp / n_gt if n_gt else None}
        result[name] = points
    return result


def fold_analysis(root, directory, fold, out, epochs):
    info, rows = evaluate.predict(root, directory, fold, mlp_epochs=epochs,
                                  save_model=out / "distance.pt", save_observations=out / "observations.npz")
    np.savez_compressed(out / "rows.npz", **rows)
    obs = dict(np.load(out / "observations.npz"))
    ids = temporal.associate(obs)
    speed = temporal.labeled_speed(root, fold["val"] + fold["test"])
    vm = np.isin(obs["seq"], fold["val"])
    validation = {k: v[vm] for k, v in obs.items()}
    # Nothing from the scored fold's test observations enters parameter selection.
    best, grid = temporal.select_filter(validation, ids[vm], speed)
    tm = np.isin(obs["seq"], fold["test"])
    test = {k: v[tm] for k, v in obs.items()}
    result, zf, vf = temporal.score_observations(test, ids[tm], speed, best["meas_rel"], best["accel"])
    np.savez_compressed(out / "tracked_test.npz", **test, predicted_track=ids[tm], filtered_z=zf, filtered_speed=vf)
    result["association"] = temporal.association_summary(test, ids[tm])
    result["selection"] = {"sequences": fold["val"], "best": best, "grid": grid}
    write_json(out / "temporal.json", clean_json(result))
    write_json(out / "evaluation.json", clean_json(info))
    write_json(out / "diagnostics.json", clean_json(diagnostics(rows)))
    write_json(out / "thresholds_validation.json", threshold_analysis(root, directory, fold, info))


def pooled_temporal(root, parts, boot):
    """Pool actual per-frame outputs, rather than averaging fold medians or percentages."""
    rows = crossval.pool(parts)
    speed = temporal.labeled_speed(root, kitti.DEVELOPMENT)
    vt = np.array([speed.get((s, int(t), int(f)), np.nan)
                   for s, t, f in zip(rows["seq"], rows["track"], rows["frame"])])
    valid = (rows["track"] >= 0) & (rows["z"] >= evaluate.MIN_Z)
    vt[~valid] = np.nan
    vr = temporal.raw_speed({**rows, "track": rows["predicted_track"]})
    vf, zf = rows["filtered_speed"], rows["filtered_z"]
    result = temporal.scores(rows, zf, vf, vt, vr)
    result["detections"] = len(vr)
    result["matched_at_least_2m"] = int(valid.sum())
    result["with_history_fraction"] = float(np.mean(np.isfinite(vr[valid])))
    result["distance_all_matched"] = {
        name: float(np.mean(np.abs(z[valid] - rows["z"][valid]) / rows["z"][valid]))
        for name, z in (("raw", rows["pred_mlp"]), ("filtered", zf))}
    result["association"] = temporal.association_summary(rows, rows["predicted_track"])
    ok = np.isfinite(vt) & np.isfinite(vr) & np.isfinite(vf)
    clusters = np.array([f"{s}/{t}" for s, t in zip(rows["seq"][ok], rows["track"][ok])])
    result["speed_mae_difference_filtered_minus_raw_95ci"] = metrics.paired_difference(
        np.abs(vf[ok] - vt[ok]), np.abs(vr[ok] - vt[ok]), clusters, boot)
    result["by_class"] = {}
    for c, name in enumerate(kitti.CLASSES):
        m = rows["gt_cls"] == c
        if (ok & m).any():
            result["by_class"][name] = temporal.scores({k: v[m] for k, v in rows.items()},
                                                       zf[m], vf[m], vt[m], vr[m])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Include all candidate NMS outputs in the signature because they determine detector selection.
    candidates = [Path(args.runs) / f"f{k}_{c}_nms" for k in range(3) for c in ("a", "b")]
    candidates += [Path(args.runs) / f"f{k}_{c}_e2e" for k in range(3) for c in ("a", "b")]
    inputs = identity(args.root, candidates + [Path(args.coco)], args.epochs, args.boot)
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    manifest = {"signature": signature, **inputs,
                "protocol": "development only; filter chosen on each model's validation predictions; automatic IoU tracks"}
    if (out / "manifest.json").exists():
        if json.loads((out / "manifest.json").read_text())["signature"] != signature:
            raise SystemExit("inputs changed; choose a new --out to preserve the old evidence")
    else:
        write_json(out / "manifest.json", manifest)
        with zipfile.ZipFile(out / "source.zip", "w") as archive:
            repo = Path(__file__).resolve().parents[1]
            for name in inputs["files"]:
                path = Path(name)
                if path.is_relative_to(repo) and path.parts[len(repo.parts)] in ("monodist", "tests", "pixi.lock", "pixi.toml", "pyproject.toml"):
                    archive.write(path, str(path.relative_to(repo)))
    table, chosen = crossval.select(args.root, args.runs, ["a", "b"])
    write_json(out / "selection.json", {"validation_map": table, "chosen": chosen})
    detectors = {"coco": [Path(args.coco)] * 3,
                 "finetuned_nms": [Path(args.runs) / f"f{k}_{chosen[k]}_nms" for k in range(3)],
                 "finetuned_e2e": [Path(args.runs) / f"f{k}_{chosen[k]}_e2e" for k in range(3)]}
    pooled = {}
    for name, directories in detectors.items():
        for k, (directory, fold) in enumerate(zip(directories, kitti.FOLDS)):
            unit = out / name / f"fold-{k}"
            unit.mkdir(parents=True, exist_ok=True)
            ready = unit / "complete.json"
            if ready.exists():
                saved = json.loads(ready.read_text())
                if saved["signature"] != signature or any(sha(unit / p) != h for p, h in saved["outputs"].items()):
                    raise SystemExit(f"invalid checkpoint: {unit}")
                print(f"reuse {unit}", flush=True)
                continue
            started = time.monotonic()
            print(f"start {unit}", flush=True)
            fold_analysis(args.root, directory, fold, unit, args.epochs)
            files = [p for p in unit.iterdir() if p.suffix in (".json", ".pt", ".npz")]
            write_json(ready, {"signature": signature, "seconds": time.monotonic() - started,
                               "outputs": {p.name: sha(p) for p in files}})
            print(f"complete {unit}: {time.monotonic() - started:.1f} s", flush=True)
        pooled[name] = crossval.pool([dict(np.load(out / name / f"fold-{k}" / "rows.npz")) for k in range(3)])
        if set(pooled[name]["seq"]) != set(kitti.DEVELOPMENT):
            raise ValueError("pooled rows must cover exactly the development sequences")
        if len(np.unique(pooled[name]["keys"])) != len(pooled[name]["keys"]):
            raise ValueError("duplicate scored object across folds")
        write_json(out / f"{name}-diagnostics.json", diagnostics(pooled[name]))
        tracked = [dict(np.load(out / name / f"fold-{k}" / "tracked_test.npz")) for k in range(3)]
        write_json(out / f"{name}-temporal.json", clean_json(pooled_temporal(args.root, tracked, args.boot)))
    for name in ("finetuned_nms", "finetuned_e2e"):
        write_json(out / f"compare-coco-vs-{name}.json", evaluate.compare(pooled["coco"], pooled[name], args.boot))
    if inputs != identity(args.root, candidates + [Path(args.coco)], args.epochs, args.boot):
        raise RuntimeError("inputs changed during execution; outputs are not accepted")
    write_json(out / "complete.json", {"signature": signature, "folds": 9, "held_out_test_evaluated": False})
    print("complete: all nine development folds; held-out test untouched", flush=True)


if __name__ == "__main__":
    main()
