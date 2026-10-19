"""Package verified measurements; numerical helpers come from the frozen measurement source."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np

ROOT = Path('/home/geminitt/projects/monodist')
SOURCE = ROOT / 'runs/source/frozen-source-2026-10-19'
DEV = ROOT / 'runs/analysis/frozen-source-development-2026-10-19'
FINAL = Path(__file__).resolve().parent
TIMING = ROOT / 'runs/kaggle/eval/frozen-source-2026-10-19'
OUT = ROOT / 'results/rebuild'
sys.path.insert(0, str(SOURCE))
from monodist import analysis, crossval, kitti, metrics

load = lambda path: json.loads(path.read_text())
assert load(FINAL / 'complete.json')['pipeline_test_evaluated']
assert load(DEV / 'complete.json')['folds'] == 9
assert load(TIMING / 'verification.json')['target_t4_measured']
OUT.mkdir(parents=True, exist_ok=True)


def distance(rows, prefix):
    clusters = np.array([f'{s}/{t}' for s,t in zip(rows['seq'],rows['track'])])
    result = {}
    for method in ('size','ground','mlp'):
        pred = rows[f'{prefix}_{method}']
        true = rows['z' if prefix == 'pred' else 'range']
        error = metrics.abs_rel(pred, true)
        result[method] = {'mean_95ci': list(map(float, metrics.cluster_bootstrap(error, clusters))),
                          'median': float(np.median(error)), 'n':len(error),'tracks':len(np.unique(clusters)),
                          'by_class':{name:{'n':int((rows['cls']==i).sum()),
                                            'mean':float(np.mean(error[rows['cls']==i])) if (rows['cls']==i).any() else None}
                                      for i,name in enumerate(kitti.CLASSES)}}
    return result


names = {'coco':'coco_nms', 'finetuned_nms':'finetuned_nms', 'finetuned_e2e':'finetuned_e2e'}
summary = {'source_commit':load(TIMING/'verification.json')['commit'], 'split':kitti.SPLIT,
           'folds':kitti.FOLDS,'development':{},'test':{},'latency':{},'selection':load(DEV/'selection.json'),
           'comparisons':{},'test_scope':'software tests may use TEST; pipeline fitting and selection use development only'}
for development_name,name in names.items():
    parts = [DEV/development_name/f'fold-{fold}' for fold in range(3)]
    rows = crossval.pool([dict(np.load(part/'rows.npz')) for part in parts])
    infos = [load(part/'evaluation.json') for part in parts]
    detection = {}
    for key in ('mAP50-95','mAP50'):
        values = [info['detection'][key] for info in infos]
        detection[key] = {'per_fold':values,'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))}
    for label,key in (('groups','detection_groups'),('agnostic','detection_agnostic')):
        values = [info[key]['mAP50-95'] for info in infos]
        detection[label] = {'per_fold':values,'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1))}
    summary['development'][name] = {'detection':detection,'distance':distance(rows,'pred'),
                                   'range':distance(rows,'range'),'temporal':load(DEV/f'{development_name}-temporal.json')}
    final = load(FINAL/name/'evaluation.json')
    summary['test'][name] = {'detection':final['detection'],'detection_groups':final['detection_groups'],
                            'detection_agnostic':final['detection_agnostic'],
                            'distance':final['distance'],'range':final['range'],
                            'recall_by_distance':final['recall_by_distance'],
                            'recall_by_distance_groups':final['recall_by_distance_groups'],
                            'recall_by_distance_agnostic':final['recall_by_distance_agnostic'],
                            'precision_by_predicted_distance':final['precision_by_predicted_distance'],
                            'temporal':load(FINAL/name/'temporal.json')}
    for suffix in ('','_pipelined'):
        timing = load(TIMING/'latency'/f'{name}{suffix}.json')
        summary['latency'][name+suffix] = {key:value for key,value in timing.items() if key!='signature'}
    print('packaged',name,flush=True)
for name in ('finetuned_nms','finetuned_e2e'):
    summary['comparisons'][f'development.{name}'] = load(DEV/f'compare-coco-vs-{name}.json')
    summary['comparisons'][f'test.{name}'] = load(FINAL/f'compare-coco-vs-{name}.json')
analysis.write_json(OUT/'summary.json',analysis.clean_json(summary))
for source, dest in ((FINAL/'freeze.json','freeze.json'),(FINAL/'manifest.json','final-manifest.json'),
                     (TIMING/'verification.json','timing-verification.json'),(DEV/'verification.json','development-verification.json')):
    shutil.copy2(source,OUT/dest)
archive = OUT/'drivers'
archive.mkdir(exist_ok=True)
for name in ('evaluate_final.py','prepare_report.py'):
    shutil.copy2(FINAL/name,archive/name)
manifest = {'measurement_source_commit':summary['source_commit'],'final_configuration':'b',
            'train_kernel':{'id':'spritker/monodist-train','version':4},
            'timing_kernel':{'id':'spritker/monodist-eval','version':8},
            'final_freeze':load(FINAL/'freeze.json'),'development_manifest':load(DEV/'manifest.json'),
            'timing_hardware_environment':load(TIMING/'verification.json'),
            'limitations':['driver paths identify the original workspace; adapt paths to reproduce elsewhere',
                           'large model/prediction artifacts remain in Kaggle output and local runs; no new public release is claimed'],
            'files':{str(path.relative_to(OUT)):analysis.sha(path) for path in OUT.rglob('*') if path.is_file() and path.name!='MANIFEST.json'}}
analysis.write_json(OUT/'MANIFEST.json',manifest)
print('report package complete',flush=True)
