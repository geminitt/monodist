import json
from pathlib import Path
import zipfile

import pytest

from monodist import bundle, geometry, kitti, mlp


def archive(tmp_path):
    """Real distance checkpoint, fake detector bytes: packaging checks only, never detector inference."""
    weights = tmp_path / 'original.pt'
    weights.write_bytes(b'packaging fixture detector')
    models = [mlp.DistanceMLP(len(geometry.FEATURES)) for _ in range(5)]
    settings = {'imgsz': '1280', 'nms': True, 'coco': False, 'half': False,
                'weights': {'name': weights.name, 'sha256': bundle.sha(weights)}}
    distance = tmp_path / 'distance.pt'
    mlp.save(distance, models, {'features':list(geometry.FEATURES), 'classes':list(kitti.CLASSES),
                               'detector':settings, 'conf':dict.fromkeys(kitti.CLASSES,0.3),
                               'temporal':{'meas_rel':0.02,'accel':1.0}})
    images = tmp_path / 'images'
    images.mkdir()
    (images / '000000.png').write_bytes(b'packaging fixture image')
    calibration = tmp_path / 'calib.txt'
    calibration.write_text('P2: 700 0 600 0 0 700 180 0 0 0 1 0\n')
    output = tmp_path / 'bundle.zip'
    bundle.build(weights,distance,images,calibration,output,frames=1)
    with zipfile.ZipFile(output) as z:
        z.extractall(tmp_path / 'relocated')
    return tmp_path / 'relocated/monodist-demo', weights, distance


def test_relocated_bundle_keeps_exact_models_and_relative_paths(tmp_path):
    root, weights, distance = archive(tmp_path)
    data = bundle.Bundle(root)
    assert bundle.sha(root/'models/detector.pt') == bundle.sha(weights)
    assert bundle.sha(root/'models/distance.pt') == bundle.sha(distance)
    assert data.camera == (700.,600.,180.)
    assert len(data.images) == 1 and data.images[0].is_relative_to(root)
    assert data.manifest['measurement_source_commit'] is None  # never invent provenance
    assert all(not Path(name).is_absolute() for name in data.manifest['files'])


def test_bundle_rejects_corrupt_bytes_and_changed_operating_point(tmp_path):
    root, _, _ = archive(tmp_path)
    config_path = root/'bundle.json'
    config = json.loads(config_path.read_text())
    config['thresholds']['Car'] = 0.9
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='thresholds/filter settings'):
        bundle.Bundle(root)
    (root/'models/detector.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='missing or changed'):
        bundle.Bundle(root)


@pytest.mark.parametrize('path',['../weights.pt','/tmp/weights.pt','C:/weights.pt','models\\weights.pt'])
def test_bundle_paths_cannot_escape_the_archive(tmp_path,path):
    with pytest.raises(ValueError,match='bundle path'):
        bundle.child(tmp_path,path)
