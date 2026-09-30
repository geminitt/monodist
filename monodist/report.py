"""Render README.md from README.template.md and the result files, so no number in the README is typed by hand.

    python -m monodist.report --write     # regenerate README.md
    python -m monodist.report --check     # exit 1 if README.md is not what the results give (run by CI)
    python -m monodist.report --summary   # every table, printed (results/summary.md)

The template holds prose with {placeholders}: {name} is a number from `values()`, {table:name} a Markdown
table from `tables()`. A placeholder without a value fails loudly. Braces right after a letter, "\\", "}", "_" or
"^" belong to LaTeX and are not placeholders.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import kitti, metrics

RESULTS = Path("results")
DETECTORS = {"coco": "COCO weights", "finetuned_nms": "fine-tuned, NMS", "finetuned_e2e": "fine-tuned, NMS-free"}
METHODS = {"size": "known size", "ground": "ground plane", "mlp": "MLP"}
CONFIG_NAMES = {"a": "a: defaults", "b": "b: lower learning rate", "c": "c: + Van, Cyclist classes",
                "d": "d: c + DontCare greyed", "e": "e: epoch chosen by NMS-free head"}


def load(path):
    return json.loads(Path(path).read_text())


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def ci(v, d=1):
    return f"{v[0] * 100:.{d}f}% [{v[1] * 100:.{d}f}, {v[2] * 100:.{d}f}]"


def table(header, rows, align=None):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align or ["---"] * len(header)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def cv_results(root=RESULTS / "cv"):
    return {k: load(root / f"{k}.json") for k in DETECTORS if (root / f"{k}.json").exists()}


def latency(root=RESULTS / "latency"):
    return {p.stem: load(p) for p in sorted(root.glob("*.json")) if not p.stem.startswith("profile")}


def values():
    """Every number the README prose uses, by name."""
    v = {}
    cv = cv_results()
    for det, r in cv.items():
        d = r["detection"]
        v[f"{det}.map"] = f"{d['mAP50-95']['mean']:.3f}"
        v[f"{det}.map_sd"] = f"{d['mAP50-95']['std']:.3f}"
        for m in METHODS:
            s = r["distance"][m]
            v[f"{det}.{m}.mean"] = pct(s["all"]["absrel_mean"][0])
            v[f"{det}.{m}.ci"] = ci(s["all"]["absrel_mean"])
            v[f"{det}.{m}.median"] = pct(s["all"]["absrel_median"][0])
            v[f"{det}.{m}.track"] = pct(s["all"]["absrel_track_mean"][0])
            v[f"{det}.{m}.metres"] = f"{s['all']['abs_m_mean']:.1f}"
            for c in kitti.CLASSES:
                v[f"{det}.{m}.{c}"] = pct(s[c]["absrel_mean"][0])
                v[f"{det}.{m}.{c}.whole"] = pct(s[f"{c} not truncated"]["absrel_mean"][0])
                for b, name in zip(("0-10", "10-20", "20-40", "40"), metrics.BIN_NAMES):
                    v[f"{det}.{m}.{c}.{b}"] = pct(s[f"{c} {name}"]["absrel_mean"][0]) if s[f"{c} {name}"].get("n") else "-"
        for c in kitti.CLASSES:
            rec = r["recall_by_distance"][c]
            v[f"{det}.recall.{c}.40"] = f"{rec['found'][3] / rec['total'][3]:.0%}"
            v[f"{det}.recall.{c}.total40"] = f"{int(rec['total'][3])}"
        v[f"{det}.n"] = f"{r['distance']['mlp']['all']['n']:,}"
        v[f"{det}.tracks"] = f"{r['distance']['mlp']['all']['tracks']}"
        v[f"{det}.floor.size.Car.whole"] = pct(r["distance_on_label_boxes"]["size"]["Car not truncated"]["absrel_mean"][0])
    for name in ("finetuned_nms", "finetuned_e2e"):
        p = RESULTS / "cv" / f"compare_coco_vs_{name}.json"
        if p.exists():
            c = load(p)
            v[f"compare.{name}.n"] = f"{c['n_common']:,}"
            for m in METHODS:
                v[f"compare.{name}.{m}"] = "{:+.1f} [{:+.1f}, {:+.1f}]".format(*(x * 100 for x in c[m]["b - a"]))
    sel = RESULTS / "cv" / "selection.json"
    if sel.exists():
        s_ = load(sel)
        v["chosen"] = ", ".join(f"fold {k}: {c}" for k, c in s_["chosen"].items())
    hyp = RESULTS / "cv" / "hypothesis_e.json"
    if hyp.exists():
        h = load(hyp)
        for c in h:
            for k2, val in h[c].items():
                v[f"hyp.{c}.{k2}"] = f"{val:.3f}" if isinstance(val, float) else str(val)
    for k in ("ft_1280", "ft_1280_fp16", "ft_1280_trt16", "ft_1280_nms", "ft_1280_nms_trt16", "ft_640", "ft_640_trt16",
              "coco_1280", "coco_1280_nms"):
        p = RESULTS / "eval" / f"{k}.json"
        if p.exists():
            v[f"fixed.{k}.map"] = f"{load(p)['detection']['mAP50-95']:.3f}"
    lat = latency()
    for k, r in lat.items():
        for s in ("decode", "preprocess", "inference", "postprocess", "total"):
            v[f"lat.{k}.{s}"] = f"{r[s]['median_ms'][0]:.1f}"
        v[f"lat.{k}.fps"] = f"{np.mean(r['fps_per_round']):.0f}"
    for head in ("e2e", "nms"):
        p = RESULTS / "latency" / f"profile_{head}.json"
        if p.exists():
            r = load(p)
            for dt in ("fp32", "fp16"):
                for key, val in r[dt].items():
                    if key == "kernels_per_forward":
                        v[f"profile.{head}.{dt}.{key}"] = f"{val:.0f}"
                    elif key == "gpu_busy_share":
                        v[f"profile.{head}.{dt}.{key}"] = pct(val, 0)
                    elif isinstance(val, float):
                        v[f"profile.{head}.{dt}.{key}"] = f"{val:.1f}"
                    elif isinstance(val, int):
                        v[f"profile.{head}.{dt}.{key}"] = str(val)
    return v


def tables():
    t = {}
    cv = cv_results()
    sel = RESULTS / "cv" / "selection.json"
    if sel.exists():
        s = load(sel)
        configs = sorted({c for f in s["val_mAP50-95"].values() for c in f})
        rows = [[f"fold {k}"] + [(f"**{f[c]:.3f}**" if s["chosen"][k] == c else f"{f[c]:.3f}") if c in f else "-" for c in configs]
                for k, f in s["val_mAP50-95"].items()]
        t["selection"] = table(["val mAP50-95", *[CONFIG_NAMES.get(c, c) for c in configs]], rows)
    if cv:
        rows = []
        for det, r in cv.items():
            d = r["detection"]
            rows.append([DETECTORS[det]] + [f"{d[m]['mean']:.3f} ± {d[m]['std']:.3f}" for m in ("Car AP50-95", "Pedestrian AP50-95", "mAP50-95", "mAP50")]
                        + [" / ".join(f"{x:.3f}" for x in d["mAP50-95"]["per_fold"])])
        t["detection"] = table(["Detector", "Car AP50-95", "Pedestrian AP50-95", "mAP50-95", "mAP50", "mAP50-95 per fold"], rows)
        rows = []
        for det, r in cv.items():
            for c in kitti.CLASSES:
                rec = r["recall_by_distance"][c]
                rows.append([DETECTORS[det], c] + [f"{f / tt:.0%} ({int(tt)})" for f, tt in zip(rec["found"], rec["total"])])
        t["recall"] = table(["Detector", "Class", *metrics.BIN_NAMES], rows)
        rows = []
        for det, r in cv.items():
            for m, mname in METHODS.items():
                s = r["distance"][m]
                rows.append([DETECTORS[det], mname, ci(s["all"]["absrel_mean"]), pct(s["all"]["absrel_median"][0]),
                             ci(s["all"]["absrel_track_mean"]), pct(s["Car not truncated"]["absrel_mean"][0]),
                             pct(s["Pedestrian not truncated"]["absrel_mean"][0])])
        t["distance"] = table(["Detector", "Method", "Mean AbsRel [95% CI]", "Median", "Per-object mean [95% CI]",
                               "Cars not truncated", "Pedestrians not truncated"], rows)
        rows = []
        for det, r in cv.items():
            for m, mname in METHODS.items():
                s = r["distance"][m]
                for c in kitti.CLASSES:
                    rows.append([DETECTORS[det], mname, c] + [f"{pct(s[f'{c} {b}']['absrel_mean'][0])} ({s[f'{c} {b}']['n']})"
                                                              if s[f"{c} {b}"].get("n") else "-" for b in metrics.BIN_NAMES])
        t["bins"] = table(["Detector", "Method", "Class", *metrics.BIN_NAMES], rows)
        rows = [[DETECTORS[det], METHODS[m]] + [pct(r["distance_on_label_boxes"][m][g]["absrel_mean"][0])
                                                for g in ("all", "Car not truncated", "Pedestrian not truncated")]
                for det, r in cv.items() for m in ("size", "ground")]
        t["floor"] = table(["Objects found by", "Method on the label boxes", "All", "Cars not truncated", "Pedestrians not truncated"], rows)
        rows = [[DETECTORS[det]] + [" / ".join(f"{f['box_iou_median'][p]:.3f}" for f in r["folds"]) for p in ("train", "val", "test")]
                for det, r in cv.items()]
        t["box_iou"] = table(["Detector", "train (folds 0/1/2)", "val", "test"], rows)
    hyp = RESULTS / "cv" / "hypothesis_e.json"
    if hyp.exists():
        h = load(hyp)
        t["hypothesis"] = table(["Fold 0", "val mAP50-95, NMS head", "val mAP50-95, NMS-free head",
                                 "test mAP50-95, NMS head", "test mAP50-95, NMS-free head", "best epoch"],
                                [[CONFIG_NAMES[c], *(f"{h[c][k]:.3f}" for k in ("val_nms", "val_e2e", "test_nms", "test_e2e")), h[c]["epoch"]]
                                 for c in ("a", "e")])
    lat = latency()
    if lat:
        stages = ("read", "decode", "preprocess", "inference", "postprocess", "overhead", "distance", "total")
        t["latency"] = table(["Configuration", *stages, "total 95% CI", "FPS per round"],
                             [[k] + [f"{r[s]['median_ms'][0]:.1f}" for s in stages]
                              + [f"{r['total']['median_ms'][1]:.1f}-{r['total']['median_ms'][2]:.1f}", "/".join(f"{x:.0f}" for x in r["fps_per_round"])]
                              for k, r in lat.items()])
    return t


def render(template):
    v, t = values(), tables()

    def sub(m):
        key = m.group(1)
        if key.startswith("table:"):
            return t[key[6:]]
        return v[key]
    # a placeholder never follows a letter, a digit, "\\", "}", "_" or "^": LaTeX groups such as \mathrm{cam} or
    # \frac{l}{2z} are left alone
    return re.sub(r"(?<![\w\\}^_])\{([a-z0-9_.:\-A-Z]+)\}", sub, template)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        print("\n\n".join(f"## {k}\n\n{v}" for k, v in tables().items()))
        return
    readme = render(Path("README.template.md").read_text())
    if args.write:
        Path("README.md").write_text(readme)
    elif Path("README.md").read_text() != readme:
        sys.exit("README.md does not match README.template.md and results/: run python -m monodist.report --write")


if __name__ == "__main__":
    main()
