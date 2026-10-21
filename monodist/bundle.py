"""Build, verify and run a portable archive of an already evaluated obstacle pipeline."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

import numpy as np

from . import geometry, kitti, mlp
from .detect import parse_imgsz
from .pipeline import Pipeline, read_p2

SCHEMA = 1


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def child(root, relative):
    """Resolve a manifest entry inside its own bundle, independent of the working directory."""
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'non-relative bundle path: {relative}')
    result = (Path(root) / str(name)).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError(f'bundle path escapes its root: {relative}')
    return result


def read(root):
    """Check the complete bundle before loading either model."""
    root = Path(root).resolve()
    manifest = json.loads((root / 'bundle.json').read_text())
    if manifest.get('schema') != SCHEMA:
        raise ValueError('unsupported bundle schema')
    for filename, digest in manifest['files'].items():
        path = child(root, filename)
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'missing or changed bundle file: {filename}')
    for name in ('weights', 'distance', 'calibration') + (('video',) if manifest.get('video') else ()):
        if manifest[name] not in manifest['files']:
            raise ValueError(f'unsigned {name}')
    if not manifest['images'] or any(name not in manifest['files'] for name in manifest['images']):
        raise ValueError('sample images must be signed')
    return manifest


class Bundle:
    """The model files are immutable; each consumer creates its own temporal pipeline state."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read(self.root)
        self.models, self.meta = mlp.load(child(self.root, self.manifest['distance']))
        geometry.feature_names(self.meta)  # refuse an ensemble on features this code cannot compute
        sig = self.meta['detector']
        if sha(child(self.root, self.manifest['weights'])) != sig['weights']['sha256']:
            raise ValueError('detector differs from the saved distance ensemble')
        if self.meta['classes'] != list(kitti.CLASSES) or len(self.models) != 5:
            raise ValueError('the bundle must use the evaluated five-class, five-model ensemble')
        if self.manifest['detector'] != {key: sig[key] for key in ('imgsz', 'nms', 'coco', 'half')}:
            raise ValueError('detector settings differ from the saved distance model')
        if self.manifest['thresholds'] != self.meta['conf'] or self.manifest['temporal'] != self.meta['temporal']:
            raise ValueError('thresholds/filter settings differ from the saved distance model')
        self.camera = read_p2(child(self.root, self.manifest['calibration']))
        self.images = [child(self.root, name) for name in self.manifest['images']]

    def pipeline(self, device='cpu'):
        os.environ.setdefault('YOLO_OFFLINE', 'true')
        settings = self.manifest['detector']
        pipe = Pipeline(child(self.root, self.manifest['weights']), child(self.root, self.manifest['distance']),
                        imgsz=parse_imgsz(settings['imgsz']), nms=settings['nms'], coco=settings['coco'],
                        half=settings['half'], temporal_settings=self.meta['temporal'])
        # Ultralytics merges model.overrides into predict arguments; keep execution explicit and offline.
        pipe.detector.overrides['device'] = device
        return pipe


def objects(result, names=kitti.CLASSES):
    """JSON-safe public outputs; infinity denotes no finite optical-axis TTC and becomes null."""
    out = []
    for i, cls in enumerate(result['cls']):
        value = {'class': names[int(cls)], 'score': float(result['score'][i]),
                 'box': list(map(float, result['box'][i])), 'depth_m': float(result['z'][i]),
                 'range_m': float(result['range'][i])}
        if 'track' in result:
            ttc = result['ttc'][i]
            value.update(track=int(result['track'][i]), filtered_depth_m=float(result['filtered_z'][i]),
                         closing_speed_mps=float(-result['speed'][i]), ttc_s=float(ttc) if np.isfinite(ttc) else None)
        out.append(value)
    return out


def build(weights, distance, images, calibration, out, frames=78, source_commit=None, video=None):
    """Copy evaluated bytes; never fit, convert, fuse, or rewrite either checkpoint."""
    weights, distance, images, calibration, out = map(Path, (weights, distance, images, calibration, out))
    if out.exists():
        raise FileExistsError(f'choose a fresh archive path: {out}')
    _, meta = mlp.load(distance)
    geometry.feature_names(meta)
    sig = meta['detector']
    if sha(weights) != sig['weights']['sha256']:
        raise ValueError('weights do not match the saved distance model')
    candidates = sorted(images.glob('*.png'))
    if not 1 <= frames <= len(candidates):
        raise ValueError(f'expected 1–{len(candidates)} sample frames')
    selected = candidates[:frames]
    numbers = [int(path.stem) for path in selected]
    if numbers != list(range(numbers[0], numbers[0] + frames)):
        raise ValueError('samples must be consecutive frames for 10 Hz tracking')
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='monodist-bundle-') as temporary:
        root = Path(temporary) / 'monodist-demo'
        for folder in ('models', 'sample/images', 'monodist'):
            (root / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights, root / 'models/detector.pt')
        shutil.copy2(distance, root / 'models/distance.pt')
        shutil.copy2(calibration, root / 'sample/calib.txt')
        if video:
            shutil.copy2(video,root/'sample/video.mp4')
        for path in selected:
            shutil.copy2(path, root / 'sample/images' / path.name)
        for path in sorted((repo / 'monodist').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.html', '.js', '.css'):
                target = root / path.relative_to(repo)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        for name in ('pixi.toml', 'pixi.lock', 'pyproject.toml', 'LICENSE'):
            shutil.copy2(repo / name, root / name)
        attribution = repo / 'data/kitti_tracking/README.md'
        if attribution.exists():
            shutil.copy2(attribution, root / 'sample/ATTRIBUTION.md')
        (root / 'START_HERE.md').write_text(
            '# monodist portable demo\n\n'
            'Requires Linux/WSL and pixi. Install dependencies once with `pixi install --frozen`; model inference is then offline.\n\n'
            'From this extracted directory:\n\n'
            '```bash\npixi run python -m monodist.bundle verify --bundle .\n'
            'pixi run python -m monodist.bundle run --bundle . --frames 6\n'
            'pixi run python -m monodist.demo --bundle .\n```\n\n'
            'Open http://127.0.0.1:8765/video for actual video inference when the bundle includes a clip. '
            'The page measures processing and display FPS, with play/pause/reset and source pacing. '
            'Open http://127.0.0.1:8765 for the 2.5D prototype. Stop the server with Ctrl+C. '
            'The simulation needs all 78 sample frames. It uses actual YOLO and distance inference on photo-billboard '
            'camera images, not a photorealistic 3D world. Braking changes future images. '
            'World truth is for rendering/diagnostics only. Nominal stationary-obstacle control was demonstrated; '
            'lead-braking and late-obstacle cases can still collide. No driving-safety claim is made.\n\n'
            'Models are the evaluated fine-tuned detector and five-seed distance ensemble. '
            'Thresholds/filter settings are saved; these commands never train or select them. '
            'Sample KITTI frames are for noncommercial use under CC BY-NC-SA 3.0; see sample/ATTRIBUTION.md. '
            'Code is AGPL-3.0; see LICENSE. Optical-axis TTC is not a collision guarantee.\n')
        manifest = {'schema': SCHEMA, 'weights': 'models/detector.pt', 'distance': 'models/distance.pt',
                    'calibration': 'sample/calib.txt', 'images': [f'sample/images/{path.name}' for path in selected],
                    'detector': {key: sig[key] for key in ('imgsz', 'nms', 'coco', 'half')},
                    'thresholds': meta['conf'], 'temporal': meta['temporal'], 'frame_period_s': 0.1,
                    'measurement_source_commit': source_commit,
                    **({'video':'sample/video.mp4'} if video else {}),
                    'build_environment': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'ultralytics')},
                    'files': {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*')) if path.is_file()}}
        (root / 'bundle.json').write_text(json.dumps(manifest, indent=1, allow_nan=False))
        Bundle(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp_archive = out.with_suffix(out.suffix + '.tmp')
        with zipfile.ZipFile(temp_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    archive.write(path, str(path.relative_to(root.parent)))
        temp_archive.replace(out)
    return {'archive': str(out), 'bytes': out.stat().st_size, 'sha256': sha(out), 'sample_frames': frames}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest='command', required=True)
    create = commands.add_parser('build')
    for name in ('weights', 'distance', 'images', 'calibration', 'out'):
        create.add_argument('--' + name, required=True)
    create.add_argument('--frames', type=int, default=78)
    create.add_argument('--video',type=Path,help='optional calibrated 10 FPS clip included in the bundle')
    create.add_argument('--source-commit', help='measurement source, if known; never inferred from current HEAD')
    for name in ('verify', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--bundle', type=Path, required=True)
        if name == 'run':
            command.add_argument('--frames', type=int, default=6)
            command.add_argument('--device', default='cpu')
    args = ap.parse_args()
    if args.command == 'build':
        print(json.dumps(build(**{key: value for key, value in vars(args).items() if key != 'command'})))
        return
    bundle = Bundle(args.bundle)
    if args.command == 'verify':
        print(json.dumps({'verified_files': len(bundle.manifest['files']), 'sample_frames': len(bundle.images),
                          'models_match': True, 'frame_period_s': bundle.manifest['frame_period_s']}))
        return
    if not 1 <= args.frames <= len(bundle.images):
        raise ValueError('frame count must fit the bundle samples')
    import torch
    torch.set_num_threads(2)
    pipe = bundle.pipeline(args.device)
    for frame, image in enumerate(bundle.images[:args.frames]):
        result = pipe([str(image)], bundle.camera)[0]
        print(json.dumps({'frame': frame, 'image': image.name, 'objects': objects(result)}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
