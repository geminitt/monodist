# KITTI tracking: labels and calibration

`training/label_02/` and `training/calib/` are the labels and camera calibration of the 21 training
sequences of the **KITTI Vision Benchmark Suite, object tracking evaluation**, unchanged. They are
included so the tests and the evaluation run on a fresh clone. The images are not included; the
Kaggle kernels read them from the mirror `leducnhuan/kitti-tracking`.

Source: A. Geiger, P. Lenz and R. Urtasun, "Are we ready for Autonomous Driving? The KITTI Vision
Benchmark Suite", CVPR 2012 — https://www.cvlibs.net/datasets/kitti/eval_tracking.php

License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0
(https://creativecommons.org/licenses/by-nc-sa/3.0/). These files are distributed under that license,
not under the repository's AGPL-3.0; they may not be used commercially.

Images of one sequence, for the study notes (78 frames, about 60 MB):

```bash
for i in $(seq -f "%06g" 0 77); do
  kaggle datasets download leducnhuan/kitti-tracking -f kitti_tracking/training/image_02/0012/$i.png \
    -p data/kitti_tracking/training/image_02/0012 -q
done
```
