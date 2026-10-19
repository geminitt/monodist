"""Score the fixed timed ensembles using frozen e10d9f3 library functions; never fit or select."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/geminitt/projects/monodist')
SOURCE = ROOT / 'runs/source/frozen-source-2026-10-19'
TRAIN = ROOT / 'runs/kaggle/train/frozen-source-2026-10-19'
EVAL = ROOT / 'runs/kaggle/eval/frozen-source-2026-10-19'
COMMIT = 'e10d9f3d6e4ade1b6571c295cf286be8b8c5c2f9'
sys.path.insert(0, str(SOURCE))
import numpy as np
from monodist import analysis, evaluate, geometry, kitti, match, metrics, mlp, temporal
from monodist.detect import merge
from monodist.pipeline import Pipeline

CONFIGS = {'coco_nms': 'coco_nms', 'finetuned_nms': 'final_b_nms', 'finetuned_e2e': 'final_b_e2e'}


def forbidden(*args, **kwargs):
    raise RuntimeError('fitting/selection is forbidden during fixed-model evaluation')


def replay(det, sizes, models, meta, seqs):
    # The detector already ran on Kaggle. Use the exact same post-detection method as the T4 timings.
    pipe = Pipeline.__new__(Pipeline)
    pipe.models, pipe.meta = models, meta
    pipe.conf = np.array([meta['conf'][name] for name in kitti.CLASSES])
    pipe.temporal_settings = meta['temporal']
    cam = evaluate.camera(ROOT / 'data/kitti_tracking', seqs, sizes)
    keep = det['score'] >= pipe.conf[det['cls']]
    indices = np.flatnonzero(keep)
    predicted = {key: np.full(len(det['score']), np.nan) for key in ('z', 'range', 'filtered_z', 'speed', 'ttc')}
    ids = np.full(len(det['score']), -1, int)
    for seq in seqs:
        pipe.reset()
        for frame in range(kitti.FRAMES[seq]):
            mask = (det['seq'] == seq) & (det['frame'] == frame)
            idx = np.flatnonzero(mask)
            raw = {key: det[key][idx] for key in ('box', 'cls', 'score')}
            raw['image_hw'] = sizes[seq]
            output = pipe.finish(raw, cam[seq][:3])
            selected = idx[keep[idx]]
            ids[selected] = output['track']
            for key in predicted:
                predicted[key][selected] = output[key]
    assert (ids[indices] >= 0).all()
    return keep, predicted, ids, cam


def score(name, seqs, out, boot):
    directory = TRAIN / 'det' / CONFIGS[name]
    calibration = EVAL / 'calibration' / name
    models, meta = mlp.load(calibration / 'distance.pt', features=geometry.FEATURES)
    assert len(models) == 5 and meta['calibration_only'] and 'test' not in meta['split']
    assert not set(meta['split']['train'] + meta['split']['val']) & set(kitti.TEST)
    assert meta['detector'] == json.loads((directory / 'run.json').read_text())
    det, sizes = merge(directory, seqs=seqs)
    keep, predicted, ids, cam = replay(det, sizes, models, meta, seqs)
    kept = {key: value[keep] for key, value in det.items()}
    # Publish actual pipeline outputs before reading scoring labels.
    np.savez_compressed(out / 'predictions.npz', **kept, **{key: value[keep] for key, value in predicted.items()},
                        predicted_track=ids[keep])
    gt = match.ground_truth(ROOT / 'data/kitti_tracking', seqs)
    status, gt_idx = match.evaluate(det, gt)
    conf = [meta['conf'][cls] for cls in kitti.CLASSES]
    thresholds = match.per_detection_conf(det, conf)
    pairs = match.matched_pairs(det, gt, status, gt_idx, thresholds)
    pair_indices = np.flatnonzero((status[:, 0] == match.TP) & keep)
    valid = pairs['z'] >= evaluate.MIN_Z
    excluded = int((~valid).sum())
    pairs = {key: value[valid] for key, value in pairs.items()}
    pair_indices = pair_indices[valid]
    za, zb, fallback = evaluate.geometric(pairs, pairs['box'], cam, meta['constants'])
    ga, gb, _ = evaluate.geometric(pairs, pairs['gt_box'], cam, meta['constants'])
    rows = {**pairs, 'pred_size': za, 'pred_ground': zb, 'pred_mlp': predicted['z'][pair_indices],
            'range_size': evaluate.to_range(pairs, za, cam), 'range_ground': evaluate.to_range(pairs, zb, cam),
            'range_mlp': predicted['range'][pair_indices], 'label_size': ga, 'label_ground': gb, 'fallback': fallback}
    rows['keys'] = np.array([f'{s}/{f}/{t}' for s, f, t in zip(rows['seq'], rows['frame'], rows['track'])])
    assert len(np.unique(rows['keys'])) == len(rows['keys'])
    np.savez_compressed(out / 'rows.npz', **rows)
    info = {'det_dir': str(directory), 'split': {'test': seqs}, 'conf': meta['conf'],
            'constants': meta['constants'], 'min_z': evaluate.MIN_Z, 'excluded_below_min_z': excluded,
            'distance_model_sha256': analysis.sha(calibration / 'distance.pt'),
            'detection': match.detection_report(det, gt, status),
            'recall_by_distance': match.recall_by_distance(det, gt, status, gt_idx, thresholds, metrics.BINS),
            **evaluate.summarize(rows, boot)}
    gd, gg = match.regroup(det, gt, kitti.GROUP_OF)
    gs, gi = match.evaluate(gd, gg)
    info['detection_groups'] = match.detection_report(gd, gg, gs, names=kitti.GROUPS)
    info['recall_by_distance_groups'] = match.recall_by_distance(gd, gg, gs, gi, thresholds, metrics.BINS, names=kitti.GROUPS)
    ad, ag = match.agnostic(det, gt)
    ast, ai = match.evaluate(ad, ag)
    info['detection_agnostic'] = match.detection_report(ad, ag, ast, names=('object',))
    info['recall_by_distance_agnostic'] = match.recall_by_distance(ad, ag, ast, ai, thresholds, metrics.BINS, names=('object',))['object']
    info['precision_by_predicted_distance'] = evaluate.precision_by_distance(kept, ast[keep], predicted['z'][keep], metrics.BINS)
    obs = evaluate.observations(kept, gt, models, cam)
    assert np.allclose(obs['pred_mlp'], predicted['z'][keep], rtol=1e-5, atol=1e-5)
    obs['pred_mlp'] = predicted['z'][keep]
    speed = temporal.labeled_speed(ROOT / 'data/kitti_tracking', seqs)
    ts, zf, vf = temporal.score_observations(obs, ids[keep], speed, **meta['temporal'])
    assert np.allclose(zf, predicted['filtered_z'][keep], rtol=1e-5, atol=1e-5)
    assert np.allclose(vf, predicted['speed'][keep], rtol=1e-5, atol=1e-5)
    ts['association'] = temporal.association_summary(obs, ids[keep])
    ts['settings'] = meta['temporal']
    vr = temporal.raw_speed({**obs, 'track': ids[keep]})
    vt = np.array([speed.get((s, int(t), int(f)), np.nan) for s, t, f in zip(obs['seq'], obs['track'], obs['frame'])])
    ok = (obs['track'] >= 0) & (obs['z'] >= evaluate.MIN_Z) & np.isfinite(vr) & np.isfinite(vt) & np.isfinite(vf)
    clusters = np.array([f'{s}/{t}' for s, t in zip(obs['seq'][ok], obs['track'][ok])])
    ts['speed_mae_difference_filtered_minus_raw_95ci'] = metrics.paired_difference(np.abs(vf[ok]-vt[ok]), np.abs(vr[ok]-vt[ok]), clusters, boot)
    np.savez_compressed(out / 'tracked.npz', **obs, predicted_track=ids[keep], filtered_z=zf, filtered_speed=vf)
    analysis.write_json(out / 'evaluation.json', analysis.clean_json(info))
    analysis.write_json(out / 'temporal.json', analysis.clean_json(ts))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE, text=True).strip()
    seqs, boot = (['0012'], 20) if args.preflight else (kitti.TEST, 2000)
    # These guards make any accidental refit/selection or access outside the scored split fail.
    mlp.train = match.f1_threshold = temporal.select_filter = forbidden
    original_labels = kitti.load_labels
    def labels(root, seq):
        assert seq in seqs, f'unexpected scoring label: {seq}'
        return original_labels(root, seq)
    kitti.load_labels = labels
    paths = list((SOURCE / 'monodist').glob('*.py')) + [Path(__file__).resolve()]
    for seq in seqs:
        paths += [ROOT / 'data/kitti_tracking/training' / kind / f'{seq}.txt' for kind in ('calib','label_02')]
    for name, directory in CONFIGS.items():
        paths += [TRAIN / 'det' / directory / 'run.json'] + [TRAIN / 'det' / directory / f'{seq}.npz' for seq in seqs]
        paths += list((EVAL / 'calibration' / name).glob('*'))
        paths += [EVAL / 'latency' / f'{name}.json']
    files = {str(path): analysis.sha(path) for path in paths if path.is_file()}
    identity = {'source_commit': COMMIT, 'scoring_driver_sha256': analysis.sha(Path(__file__)), 'files': files,
                'sequences': seqs, 'bootstrap_samples': boot, 'preflight': args.preflight,
                'final_configuration': 'b', 'environment': {key: importlib.metadata.version(key) for key in ('numpy','torch','ultralytics')}}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / 'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())['signature'] == signature, 'inputs changed; use another output folder'
    else:
        analysis.write_json(manifest, dict(signature=signature, **identity))
    pooled = {}
    for name in CONFIGS:
        unit = args.out / name
        unit.mkdir(exist_ok=True)
        complete = unit / 'complete.json'
        if complete.exists():
            saved = json.loads(complete.read_text())
            assert saved['signature'] == signature and all(analysis.sha(unit/key)==value for key,value in saved['outputs'].items())
            print('reuse', name, flush=True)
        else:
            started = time.monotonic()
            print('start', name, flush=True)
            score(name, seqs, unit, boot)
            analysis.write_json(complete, {'signature':signature,'seconds':time.monotonic()-started,
                                          'outputs':{path.name:analysis.sha(path) for path in unit.iterdir() if path.is_file()}})
            print('complete', name, flush=True)
        pooled[name] = dict(np.load(unit / 'rows.npz'))
    for name in ('finetuned_nms','finetuned_e2e'):
        analysis.write_json(args.out / f'compare-coco-vs-{name}.json', analysis.clean_json(evaluate.compare(pooled['coco_nms'], pooled[name], boot)))
    assert all(analysis.sha(path)==value for path,value in files.items()), 'inputs changed during scoring'
    analysis.write_json(args.out / 'complete.json', {'signature':signature,'configurations':3,'pipeline_test_evaluated':not args.preflight})
    print('complete fixed-model evaluation; no fitting or selection', flush=True)


if __name__ == '__main__':
    main()
