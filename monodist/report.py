"""Collect results/eval/*.json and results/latency/*.json into one Markdown summary.

    python -m monodist.report > results/summary.md
"""
import json
import sys
from pathlib import Path

from . import kitti, metrics

METHODS = {"size": "known size", "ground": "ground plane", "mlp": "MLP"}


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))}


def ci(v, digits=3, pct=True):
    k = 100 if pct else 1
    return f"{v[0] * k:.1f}% [{v[1] * k:.1f}, {v[2] * k:.1f}]" if pct else f"{v[0]:.{digits}f} [{v[1]:.{digits}f}, {v[2]:.{digits}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(eval_dir="results/eval", latency_dir="results/latency"):
    ev = {k: v for k, v in load(eval_dir).items() if not k.startswith("compare_")}
    cmp = {k: v for k, v in load(eval_dir).items() if k.startswith("compare_")}
    lat = load(latency_dir) if Path(latency_dir).is_dir() else {}
    w = sys.stdout.write

    w("## Detection on the test sequences\n\n")
    w(table(["Configuration", "Car AP50-95", "Pedestrian AP50-95", "mAP50-95", "mAP50"],
            [[k, f"{r['detection']['Car']['AP50-95']:.3f}", f"{r['detection']['Pedestrian']['AP50-95']:.3f}",
              f"{r['detection']['mAP50-95']:.3f}", f"{r['detection']['mAP50']:.3f}"] for k, r in ev.items()]) + "\n\n")

    w("## Operating point: per-class score threshold with the best F1 on the val sequences\n\n")
    w(table(["Configuration", *kitti.CLASSES], [[k] + [f"{r['conf'][c]:.3f}" for c in kitti.CLASSES] for k, r in ev.items()]) + "\n\n")
    w("## Recall by distance at that operating point (IoU >= 0.5)\n\n")
    rows = []
    for k, r in ev.items():
        for c in kitti.CLASSES:
            rec = r["recall_by_distance"][c]
            rows.append([k, c] + [f"{f / t:.0%} ({int(t)})" if t else "-" for f, t in zip(rec["found"], rec["total"])])
    w(table(["Configuration", "Class", *metrics.BIN_NAMES], rows) + "\n\n")

    w("## Distance error (AbsRel = |Zhat - Z| / Z), mean with 95% cluster-bootstrap interval, median in brackets\n\n")
    rows = []
    for k, r in ev.items():
        for m, name in METHODS.items():
            d = r["distance"][m]
            rows.append([k, name] + [f"{ci(d[g]['absrel_mean'])} ({d[g]['absrel_median'][0]:.1%})" if d[g].get("n") else "-"
                                     for g in ("all", "Car", "Pedestrian", "Car not truncated", "Pedestrian not truncated")])
    w(table(["Detector", "Method", "All", "Car", "Pedestrian", "Car, not truncated", "Pedestrian, not truncated"], rows)
      + "\n\n")

    w("## Distance error by distance bin (mean AbsRel)\n\n")
    rows = []
    for k, r in ev.items():
        for m, name in METHODS.items():
            d = r["distance"][m]
            for c in kitti.CLASSES:
                rows.append([k, name, c] + [f"{d[f'{c} {b}']['absrel_mean'][0]:.1%} (n={d[f'{c} {b}']['n']})"
                                            if d[f"{c} {b}"].get("n") else "-" for b in metrics.BIN_NAMES])
    w(table(["Detector", "Method", "Class", *metrics.BIN_NAMES], rows) + "\n\n")

    w("## Paired differences between methods (mean AbsRel of the first minus the second, same objects)\n\n")
    rows = []
    for k, r in ev.items():
        for pair, v in r["distance"]["paired_absrel_difference"].items():
            rows.append([k, pair] + [f"{v[g][0] * 100:+.1f} [{v[g][1] * 100:+.1f}, {v[g][2] * 100:+.1f}]" for g in ("all", *kitti.CLASSES)])
    w(table(["Detector", "Methods", "All (points)", "Car", "Pedestrian"], rows) + "\n\n")

    w("## Geometric methods on the labelled boxes of the same objects (error floor with a perfect detector)\n\n")
    rows = [[k, METHODS[m]] + [f"{r['distance_on_label_boxes'][m][g]['absrel_mean'][0]:.1%}" for g in ("all", *kitti.CLASSES)]
            for k, r in ev.items() for m in ("size", "ground")]
    w(table(["Objects matched by", "Method", "All", "Car", "Pedestrian"], rows) + "\n\n")

    w("## Detector boxes vs labels: median IoU of matched boxes per split\n\n")
    w(table(["Configuration", *kitti.SPLIT], [[k] + [f"{r['box_iou_median'][s]:.3f}" for s in kitti.SPLIT] for k, r in ev.items()])
      + "\n\n")

    if cmp:
        w("## Detectors compared on the objects both found (mean AbsRel, second minus first)\n\n")
        rows = []
        for k, c in cmp.items():
            for m, name in METHODS.items():
                rows.append([k.removeprefix("compare_"), name, c["n_common"], f"{c[m]['a'][0]:.1%}", f"{c[m]['b'][0]:.1%}",
                             f"{c[m]['b - a'][0] * 100:+.1f} [{c[m]['b - a'][1] * 100:+.1f}, {c[m]['b - a'][2] * 100:+.1f}]"])
        w(table(["Comparison", "Method", "Objects", "First", "Second", "Difference (points)"], rows) + "\n\n")

    if lat:
        w("## Latency per frame on a Kaggle T4 (median ms over frames x rounds, 95% bootstrap interval of the median)\n\n")
        stages = ("read", "decode", "preprocess", "inference", "postprocess", "overhead", "distance", "total")
        rows = [[k] + [f"{v[s]['median_ms'][0]:.1f}" for s in stages] + [f"{v['total']['median_ms'][1]:.1f}-{v['total']['median_ms'][2]:.1f}",
                                                                          "/".join(f"{x:.0f}" for x in v["fps_per_round"])]
                for k, v in lat.items()]
        w(table(["Configuration", *stages, "total CI", "FPS per round"], rows) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
