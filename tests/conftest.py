from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "data" / "kitti_tracking"


@pytest.fixture
def root():
    if not (ROOT / "training" / "label_02").is_dir():
        pytest.skip("KITTI labels not downloaded (data/kitti_tracking/training/label_02)")
    return ROOT
