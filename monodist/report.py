"""Render the rebuilt README from results/rebuild; never mix legacy measurements."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys

RESULTS = Path('results/rebuild')
DETECTORS = {'coco_nms': 'COCO NMS', 'finetuned_nms': 'Fine-tuned NMS', 'finetuned_e2e': 'Fine-tuned NMS-free'}
METHODS = {'size': 'Known size', 'ground': 'Ground plane', 'mlp': 'MLP ensemble'}


def load():
    return json.loads((RESULTS / 'summary.json').read_text())


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def interval(values):
    return '—' if values is None else f'{100*values[0]:.2f}% [{100*values[1]:.2f}, {100*values[2]:.2f}]'


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def values():
    data = load()
    model = data['test']['finetuned_nms']
    return {'source': data['source_commit'], 'chosen': 'lower learning rate (configuration b)',
            'test.nms.map': f"{model['detection']['mAP50-95']:.3f}",
            'test.nms.distance': pct(model['distance']['mlp']['all']['absrel_mean'][0]),
            'test.nms.speed.raw': f"{model['temporal']['raw']['speed_mae']:.2f}",
            'test.nms.speed.filtered': f"{model['temporal']['filtered']['speed_mae']:.2f}",
            'lat.nms.serial': f"{data['latency']['finetuned_nms']['total']['median_ms'][0]:.2f}",
            'lat.nms.fps': f"{statistics.median(data['latency']['finetuned_nms_pipelined']['fps_per_round']):.1f}"}


def distance_stat(model, partition, target, method):
    value = model[target][method]
    if partition == 'development':
        return value['mean_95ci'], value['median'], value['n'], value['tracks']
    return value['all']['absrel_mean'], value['all']['absrel_median'][0], value['all']['n'], value['all']['tracks']


def tables():
    data, out = load(), {}
    selection = data['selection']
    out['selection'] = table(['Development fold', 'Defaults: validation AP', 'Lower LR: validation AP', 'Chosen'],
        [[fold, f"{row['a']:.3f}", f"{row['b']:.3f}", selection['chosen'][fold]]
         for fold, row in selection['validation_map'].items()])
    detection, distance, temporal, per_class, recall = [], [], [], [], []
    for partition in ('development', 'test'):
        for name, model in data[partition].items():
            d = model['detection']
            if partition == 'development':
                ap = [f"{d[key]['mean']:.3f} ± {d[key]['std']:.3f}" for key in ('mAP50-95','mAP50','groups','agnostic')]
            else:
                ap = [f"{d[key]:.3f}" for key in ('mAP50-95','mAP50')]
                ap += [f"{model[key]['mAP50-95']:.3f}" for key in ('detection_groups','detection_agnostic')]
            detection.append([partition, DETECTORS[name], *ap])
            for method, label in METHODS.items():
                z, median, n, tracks = distance_stat(model, partition, 'distance', method)
                rng, _, _, _ = distance_stat(model, partition, 'range', method)
                distance.append([partition, DETECTORS[name], label, interval(z), pct(median), interval(rng), n, tracks])
            t = model['temporal']
            temporal.append([partition, DETECTORS[name], t['n'],
                f"{t['raw']['speed_mae']:.2f} → {t['filtered']['speed_mae']:.2f}",
                f"{pct(t['raw']['ttc_within_20pct'])} → {pct(t['filtered']['ttc_within_20pct'])}",
                t['ttc_n'], pct(t['with_history_fraction'])])
            if partition == 'test':
                for cls in ('Car','Van','LargeVehicle','Pedestrian','Cyclist'):
                    z, rng = model['distance']['mlp'][cls], model['range']['mlp'][cls]
                    per_class.append([DETECTORS[name], cls, z['n'], z['tracks'],
                                      interval(z.get('absrel_mean')), interval(rng.get('absrel_mean'))])
                    counts = model['recall_by_distance'][cls]
                    recall.append([DETECTORS[name], cls, *[f'{found/total:.1%} ({int(total)})' if total else '—'
                        for found,total in zip(counts['found'],counts['total'])]])
    out['detection'] = table(['Partition','Detector','5-class AP50–95','AP50','3-group AP50–95','Class-agnostic AP50–95'], detection)
    out['distance'] = table(['Partition','Detector','Method','Mean depth AbsRel [95% CI]','Median depth AbsRel',
                            'Mean range AbsRel [95% CI]','Object-frames','Tracks'], distance)
    out['temporal'] = table(['Partition','Detector','Speed rows','Speed MAE raw → filtered (m/s)',
                            'TTC within 20%, raw → filtered','TTC rows','Matched rows with history'], temporal)
    out['classes'] = table(['Detector','Class','Object-frames','Tracks','Depth AbsRel [95% CI]','Range AbsRel [95% CI]'], per_class)
    out['recall'] = table(['Detector','Class','0–10 m','10–20 m','20–40 m','>40 m'], recall)
    latency = []
    for name,label in DETECTORS.items():
        serial, overlap = data['latency'][name], data['latency'][name+'_pipelined']
        latency.append([label, *[f"{serial[key]['median_ms'][0]:.2f}" for key in ('decode','inference','distance','total')],
            f"{statistics.median(serial['fps_per_round']):.1f}", f"{statistics.median(overlap['fps_per_round']):.1f}"])
    out['latency'] = table(['Pipeline','Decode (ms)','Detector forward (ms)','Distance/tracking/TTC (ms)',
                           'Serial total (ms)','Serial FPS','Decode-overlapped FPS'], latency)
    comparisons = []
    for key,value in data['comparisons'].items():
        partition,name = key.split('.')
        difference = value['mlp']['b - a']
        comparisons.append([partition, DETECTORS[name], value['n_common'], value['tracks'],
            '—' if difference is None else '{:+.2f} [{:+.2f}, {:+.2f}]'.format(*(100*v for v in difference))])
    out['paired'] = table(['Partition','Compared with COCO NMS','Common object-frames','Tracks',
                          'MLP AbsRel difference [95% CI], percentage points'], comparisons)
    return out


def render(template):
    v, t = values(), tables()
    def sub(match):
        key = match.group(1)
        return t[key[6:]] if key.startswith('table:') else v[key]
    return re.sub(r'(?<![\w\\}^_])\{([a-z0-9_.:\-A-Z]+)\}', sub, template)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--write', action='store_true')
    group.add_argument('--check', action='store_true')
    group.add_argument('--summary', action='store_true')
    args = ap.parse_args()
    if args.summary:
        print('\n\n'.join(f'## {name}\n\n{text}' for name,text in tables().items()))
        return
    readme = render(Path('README.template.md').read_text())
    if args.write:
        Path('README.md').write_text(readme)
    elif Path('README.md').read_text() != readme:
        sys.exit('README.md differs from the verified records/template; run python -m monodist.report --write')


if __name__ == '__main__':
    main()
