"""Fit and save the final distance/filter configuration using development data only.

This is calibration, not held-out test evaluation. Validation rows select confidence and filter settings;
no labels or detection files from kitti.TEST are loaded.
"""
import argparse
from pathlib import Path

import numpy as np

from . import analysis, evaluate, kitti, mlp, temporal


def fit(root, detections, out, epochs=150, split=None):
    split = split or kitti.SPLIT
    if set(split["train"] + split["val"]) & set(kitti.TEST):
        raise ValueError("calibration may use development sequences only")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # predict's scored partition is validation here, solely to obtain validation diagnostics with the fitted
    # model. The saved model below explicitly records that no held-out evaluation took place.
    validation_split = {"train": split["train"], "val": split["val"], "test": split["val"]}
    info, rows = evaluate.predict(root, detections, validation_split, mlp_epochs=epochs,
                                  save_model=out / "distance.pt", save_observations=out / "validation.npz")
    obs = dict(np.load(out / "validation.npz"))
    ids = temporal.associate(obs)
    speed = temporal.labeled_speed(root, split["val"])
    best, grid = temporal.select_filter(obs, ids, speed)
    settings = {k: best[k] for k in ("meas_rel", "accel")}
    models, meta = mlp.load(out / "distance.pt")
    meta.update(split={"train": split["train"], "val": split["val"]}, calibration_only=True, temporal=settings)
    mlp.save(out / "distance.pt", models, meta)
    analysis.write_json(out / "temporal.json", settings)
    analysis.write_json(out / "selection.json", {"validation_sequences": split["val"], "best": best, "grid": grid})
    analysis.write_json(out / "validation-diagnostics.json", analysis.diagnostics(rows))
    analysis.write_json(out / "calibration.json", {"held_out_test_evaluated": False, "thresholds": info["conf"],
                                                  "temporal": settings, "train_sequences": split["train"],
                                                  "validation_sequences": split["val"], "mlp": info["mlp"]})
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = analysis.identity(args.root, [args.det], args.epochs, 0)
    manifest = out / "manifest.json"
    if manifest.exists():
        import json
        if json.loads(manifest.read_text()) != identity:
            raise SystemExit("calibration inputs changed; use a new --out")
        complete = out / "complete.json"
        if complete.exists():
            outputs = json.loads(complete.read_text())["outputs"]
            if any(analysis.sha(out / name) != digest for name, digest in outputs.items()):
                raise SystemExit("calibration checkpoint is corrupt")
            print(f"reuse complete calibration: {out}", flush=True)
            return
    else:
        analysis.write_json(manifest, identity)
    fit(args.root, args.det, out, args.epochs)
    if identity != analysis.identity(args.root, [args.det], args.epochs, 0):
        raise RuntimeError("calibration inputs changed during execution")
    analysis.write_json(out / "complete.json", {"outputs": {p.name: analysis.sha(p) for p in out.iterdir()
                                                           if p.suffix in (".npz", ".pt", ".json") and p.name != "complete.json"}})
    print(f"calibration complete: {out}; held-out test not evaluated", flush=True)


if __name__ == "__main__":
    main()
