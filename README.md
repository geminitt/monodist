<div align="center">

# monodist

[![CI](https://img.shields.io/github/actions/workflow/status/geminitt/monodist/ci.yml?branch=main&label=CI&style=for-the-badge)](https://github.com/geminitt/monodist/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/PYTHON-3.12-A19654?style=for-the-badge)](./pixi.toml)
[![License](https://img.shields.io/badge/LICENSE-AGPL--3.0-6B7F4E?style=for-the-badge)](./LICENSE)

**Detect cars and pedestrians in KITTI video with YOLO26n, estimate how far away they are from a single
camera, and measure what each stage of the pipeline costs on a Kaggle T4.**

</div>

<!-- README.md is generated from README.template.md and results/ by `python -m monodist.report --write`;
     CI fails when they disagree. Edit the template, never README.md. -->

---

## Result in one paragraph

Over a three-fold cross-validation that tests every one of the 21 labelled KITTI sequences once, a
5k-parameter MLP that reads only the detected box estimates distance with a **mean relative error of
8.1% [7.6, 8.6]** (median 5.4%), using YOLO26n with its COCO weights and no detector training at
all. The textbook pinhole formula $Z = fH/h$ gets 12.6% and the flat-road formula
24.0%. Fine-tuning the detector on KITTI raises mAP50-95 from 0.395 ± 0.061 to
**0.464 ± 0.024**, but it does not make the distances better
(8.9% [8.0, 10.1]; the same 9.6% as COCO when every object counts once). On the
T4, decoding the PNG costs more than running the network; TensorRT fp16 cuts inference from
9.2 to 3.8 ms at the same accuracy, and decoding the
next frame on a worker thread takes the fine-tuned detector with NMS to **80
frames per second**.

---

## What is compared

| Axis | Options |
|---|---|
| Detector | YOLO26n with its COCO weights (never trained); YOLO26n fine-tuned on KITTI, per fold |
| Fine-tuning configuration, chosen per fold on val | a: defaults · b: lower learning rate · c: Van and Cyclist labelled as their own classes · d: c with the DontCare regions painted grey |
| YOLO26 head | one-to-many head followed by NMS; NMS-free (one-to-one) head |
| Distance from a box | **known size** $Z = fH/h$ ($H$ = mean class height); **ground plane** $Z = f\,h_\mathrm{cam}/(v_2 - c_y)$ (bottom edge on a flat road); **MLP** trained on the detector's own boxes |
| Speed levers | fp16, TensorRT fp16, CUDA graphs, input 1280 vs 640, decoding the next frame on a worker thread |

---

## Data and protocol

- **KITTI tracking**, the 21 labelled sequences (8,008 frames), classes Car and Pedestrian; the official test
  sequences have no public labels. Ground truth is the depth z of the LiDAR-based 3D box in camera 2, with
  the calibration of each sequence (four calibrations, $f$ = 707 to 722 px).
- **Three-fold cross-validation by sequence** (`kitti.FOLDS`): each sequence is test exactly once; the other
  sequences of a fold split about 70/30 into train and val. Folds balance cars (instances and tracks) and
  pedestrian tracks; pedestrian instances cannot be balanced, since sequence 0019 alone holds 53% of them.
- **Per fold**: configurations a-d are fine-tuned on train (epoch chosen by Ultralytics on val; for c and d
  its val score also averages the Van and Cyclist classes) and the one with the best val mAP50-95 on Car and
  Pedestrian, scored with the project's own matcher, is kept (fold 0: b, fold 1: b, fold 2: a). The known-size
  heights and the camera height come from the train labels; the MLP trains on the detector's boxes on the
  train sequences (epochs by leave-one-sequence-out, five seeds averaged); the score threshold of each class
  is the F1-best one on val. Nothing is chosen on test.
- **Matching**: greedy by score, $\mathrm{IoU} \ge 0.5$, per class; as in the KITTI devkit a car detection on a Van and a
  pedestrian detection on a sitting Person count as neither hit nor false alarm, and detections inside
  DontCare regions are ignored. mAP is COCO-style (101-point, $\mathrm{IoU} = 0.50, 0.55, \dots, 0.95$), per fold, reported as mean ±
  standard deviation over the three folds.
- **Distance metric**: $\mathrm{AbsRel} = \lvert \hat Z - Z \rvert / Z$ on matched test objects farther than 2 m, pooled over the
  folds. Intervals are **cluster bootstrap over tracks**: a car seen in 300 frames is one sample, not 300.
  Next to the usual per-frame mean, the per-object mean gives every tracked object the same weight.

---

## Results

All tables, generated from `results/`, are also in [results/summary.md](results/summary.md).

### Choice of the fine-tuning configuration (val mAP50-95 per fold; bold = kept)

| val mAP50-95 | a: defaults | b: lower learning rate | c: + Van, Cyclist classes | d: c + DontCare greyed |
|---|---|---|---|---|
| fold 0 | 0.464 | **0.476** | 0.446 | 0.444 |
| fold 1 | 0.505 | **0.521** | 0.512 | 0.503 |
| fold 2 | **0.531** | 0.524 | 0.516 | 0.527 |

Labelling vans and cyclists, with or without greying DontCare regions, never won a fold (c came second in
fold 1); a lower learning rate won two folds by about a point over the defaults.

### Detection (test, per fold, mean ± standard deviation over folds)

| Detector | Car AP50-95 | Pedestrian AP50-95 | mAP50-95 | mAP50 | mAP50-95 per fold |
|---|---|---|---|---|---|
| COCO weights | 0.503 ± 0.040 | 0.287 ± 0.105 | 0.395 ± 0.061 | 0.746 ± 0.056 | 0.462 / 0.379 / 0.344 |
| fine-tuned, NMS | 0.614 ± 0.033 | 0.315 ± 0.023 | 0.464 ± 0.024 | 0.790 ± 0.031 | 0.490 / 0.460 / 0.443 |
| fine-tuned, NMS-free | 0.574 ± 0.039 | 0.283 ± 0.023 | 0.429 ± 0.029 | 0.723 ± 0.027 | 0.455 / 0.434 / 0.397 |

### Distance error on the test objects of all folds

| Detector | Method | Mean AbsRel [95% CI] | Median | Per-object mean [95% CI] | Cars not truncated | Pedestrians not truncated |
|---|---|---|---|---|---|---|
| COCO weights | known size | 12.6% [10.7, 15.5] | 7.5% | 13.0% [12.1, 14.0] | 8.7% | 8.4% |
| COCO weights | ground plane | 24.0% [21.6, 26.7] | 15.8% | 28.7% [24.8, 33.3] | 24.3% | 16.9% |
| COCO weights | MLP | 8.1% [7.6, 8.6] | 5.4% | 9.6% [9.1, 10.1] | 6.9% | 7.0% |
| fine-tuned, NMS | known size | 12.6% [11.2, 14.4] | 8.7% | 12.6% [11.8, 13.4] | 10.6% | 7.1% |
| fine-tuned, NMS | ground plane | 24.6% [22.0, 27.4] | 16.1% | 29.7% [25.1, 35.6] | 24.9% | 14.8% |
| fine-tuned, NMS | MLP | 8.9% [8.0, 10.1] | 5.5% | 9.6% [9.0, 10.2] | 6.8% | 6.9% |
| fine-tuned, NMS-free | known size | 12.9% [11.4, 15.0] | 9.3% | 12.8% [12.1, 13.6] | 10.9% | 7.2% |
| fine-tuned, NMS-free | ground plane | 24.8% [21.9, 27.8] | 16.1% | 30.5% [25.3, 36.8] | 25.0% | 14.5% |
| fine-tuned, NMS-free | MLP | 9.0% [8.0, 10.3] | 5.3% | 9.7% [9.2, 10.4] | 6.7% | 6.6% |

Test set: 27,929 object-frames from 722 tracks for the COCO detector.

### Recall by distance at the operating point ($\mathrm{IoU} \ge 0.5$)

| Detector | Class | 0-10 m | 10-20 m | 20-40 m | >40 m |
|---|---|---|---|---|---|
| COCO weights | Car | 90% (3264) | 87% (5464) | 77% (11853) | 61% (6718) |
| COCO weights | Pedestrian | 75% (3901) | 66% (5164) | 36% (2305) | 1% (100) |
| fine-tuned, NMS | Car | 93% (3264) | 90% (5464) | 82% (11853) | 64% (6718) |
| fine-tuned, NMS | Pedestrian | 80% (3901) | 60% (5164) | 23% (2305) | 0% (100) |
| fine-tuned, NMS-free | Car | 89% (3264) | 83% (5464) | 74% (11853) | 56% (6718) |
| fine-tuned, NMS-free | Pedestrian | 65% (3901) | 56% (5164) | 17% (2305) | 0% (100) |

### Latency

One Kaggle session (T4, 4 vCPU Xeon), one frame at a time, 300 frames of sequence 0007 x 3 rounds after 30
warm-up frames; medians in ms. "Pipelined" decodes the next frame on a worker thread while the GPU works on
the current one: its FPS is throughput, the latency of one frame stays about the serial total. FPS is the
mean of the three rounds (the session's first configuration ran its first round at 24 FPS, a cold start;
every other round is within 5 FPS of its configuration's mean). The accuracy
column is mAP50-95 of the same weights on the earlier fixed split (see `results/eval/`).

| Configuration | mAP50-95 | decode | preprocess | inference | postprocess | total, serial | FPS serial | FPS pipelined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO weights, PyTorch fp32, 1280 | 0.413 | 12.0 | 2.6 | 9.8 | 0.6 | 27.6 | 33 | |
| fine-tuned, PyTorch fp32, 1280 | 0.451 | 12.1 | 3.5 | 9.6 | 0.7 | 27.3 | 37 | 75 |
| fine-tuned, PyTorch fp16, 1280 | 0.451 | 11.2 | 2.3 | 9.1 | 0.5 | 24.7 | 40 | |
| fine-tuned, torch.compile (CUDA graphs), 1280 | | 11.1 | 2.0 | 9.3 | 0.5 | 24.6 | 41 | |
| fine-tuned + NMS, PyTorch fp32, 1280 | 0.477 | 11.8 | 3.3 | 9.2 | 1.4 | 27.4 | 37 | 74 |
| fine-tuned, TensorRT fp16, 1280 | 0.451 | 11.1 | 2.3 | 3.9 | 0.5 | 19.4 | 51 | 77 |
| **fine-tuned + NMS, TensorRT fp16, 1280** | **0.476** | 11.0 | 2.3 | 3.8 | 1.2 | 20.1 | 49 | **80** |
| fine-tuned, PyTorch fp32, 640 | 0.358 | 11.1 | 0.8 | 8.2 | 0.5 | 22.4 | 44 | |
| fine-tuned, TensorRT fp16, 640 | 0.358 | 11.1 | 0.8 | 2.7 | 0.5 | 16.7 | 59 | 79 |

---

## Findings

1. **The box already contains most of the distance.** The MLP beats both formulas for both classes: for
   cars at 10-20 m it scores 6.3% against 9.8% for the known-size formula,
   and for cars closer than 10 m, most of them cut by the image border, 16.9% against
   46.8%.
2. **The known-size formula has a floor of about 10% on cars, even with perfect boxes.** KITTI measures depth
   to the centre of the car, but a 2D box encloses the whole car: its bottom edge comes from the nearest face
   and its top edge from the far end of the roof. For a car seen from behind, with the ground $y$ metres below
   the camera, $\hat Z / z \approx 1 / \left(1 + \frac{l}{2z} \cdot \frac{2y - H}{H}\right)$, about 10% low at 25 m for a 4.3 m car,
   which matches the labels (study notes, notebook 01). On the labelled boxes the formula scores
   10.1% on untruncated cars. A detector whose boxes look *more* like the labels makes
   it *worse*: 10.6% after fine-tuning against 8.7% with the COCO
   weights.
3. **The flat-road formula does not hold on KITTI.** The labelled ground points of cars lie between 1.38 and
   1.80 m below the camera (interquartile range), because of slopes and pitch; pedestrians stand about 16 cm
   higher, on the pavement; and half a degree of pitch moves the horizon by 6 px, a 23% error at 40 m.
4. **Fine-tuning helps detection, not distance.** On the 25,436 test object-frames both
   detectors found, the fine-tuned MLP differs by +0.7 [+0.0, +1.7] points of mean AbsRel; one car
   of sequence 0015 that stays 2-5 m from the camera, cut by the border, accounts for most of it, and the
   medians (5.4%, 5.5%) and per-object means (9.6%,
   9.6%) agree.
5. **After fine-tuning, YOLO26's NMS-free head lags behind its NMS head** (0.429 vs
   0.464), while with the COCO weights the two are equal on the fixed split
   (0.413 vs 0.414). *Hypothesis tested and rejected:* the lag is not
   caused by choosing the epoch with the NMS head. Configuration e chose the epoch with the NMS-free head;
   on fold 0 it picked the same epoch (13), and the NMS-free head stays behind at every epoch of
   training (test 0.455 vs 0.473).
6. **Eager PyTorch is partly limited by kernel launches.** One forward pass launches
   400 CUDA kernels in fp32 (480 in
   fp16). Replaying them as one CUDA graph cuts the forward pass from 10.3 to
   5.5 ms in fp32 and from 11.3 to
   3.3 ms in fp16: the time eager PyTorch loses between kernels is what hides
   fp16's speed. Ultralytics' `compile="reduce-overhead"` did not capture that gain
   (9.3 ms), so the practical lever remains TensorRT.
7. **Far objects are the weak point.** Beyond 40 m the COCO model finds 61% of the cars and
   1% of the 100 pedestrian object-frames; input 640
   instead of 1280 loses more far objects (fixed split).

---

## Checks

- **AP implementation**: on the fixed split's val sequences, without the ignore rules and with NMS, it gives
  mAP50-95 0.466 (Car 0.548, Pedestrian 0.384) against 0.465 (0.552, 0.378) from Ultralytics' own
  validation of the same weights.
- **Labels**: the 3D boxes projected with P2 overlap the 2D labels at a median IoU of 0.97.
- **Formulas**: both invert the pinhole projection exactly on synthetic boxes (unit tests).
- **Reproducibility**: GPU and CPU detections agree on 46,230 of 46,231 boxes; configuration e, trained with
  the same seed and stopped at the same epoch as a, reproduced a's detections byte for byte; rerunning the
  evaluation reproduces its JSON byte for byte.
- **Precision**: PyTorch fp32, fp16 and TensorRT fp16 give the same mAP50-95 to three decimals.
- **Every number in this README** is filled in from `results/` by `monodist/report.py`, and CI checks it.

---

## Limitations

- One fine-tuning run per fold and configuration; the spread over folds mixes data and seed variation.
- Pedestrians beyond 40 m are almost absent from KITTI tracking (100
  object-frames), and no detector finds them.
- The mean AbsRel is sensitive to a few objects seen for hundreds of frames; the per-object mean and the
  median are reported next to it.
- Latency is for one frame at a time on a Kaggle T4 with 4 CPU cores; PNG decoding depends on the CPU. The
  latency session used the fixed-split fine-tuned weights (same architecture as the fold models).
- GPU code paths (detection, fine-tuning, latency) are tested by running them on Kaggle, not by CI.

---

## Mistakes caught while doing this

Each one was found by checking a number that looked too good or too odd, and fixed before the results above.

1. **The distance MLP first trained on the val sequences**, where 91% of the pedestrians came from one
   sequence; it was worse than the plain formula on pedestrians. It now trains on the train sequences.
2. **Both TensorRT engines were written to the same file**, so a "1280" timing measured the 640 engine; the
   latency tool now refuses to run an engine at a shape other than the one asked for.
3. **A fixed score threshold of 0.25** favoured whichever detector outputs higher scores; each detector now
   uses its F1-best threshold from val.
4. **One MLP seed** moved the error by about a point; the prediction is now the average of five seeds.
5. **Objects closer than 2 m** (all truncated) made AbsRel explode; they are excluded and counted.
6. **Latency from different Kaggle sessions** is not comparable; all timings are from one session.
7. **A test set of 8 sequences** held only 30 pedestrian tracks and one run of fine-tuning; the
   cross-validation now tests all 21 sequences and every configuration three times.

---

## Reproduce

```bash
pixi install && pixi run test               # CPU tests, KITTI labels included in the repository

# Kaggle (GPU T4, dataset leducnhuan/kitti-tracking), code pinned to the pushed HEAD:
python kaggle/queue.py --commit $(git rev-parse HEAD)   # 13 fine-tuning jobs, two at a time, ~6 h of GPU
python kaggle/push.py eval --mode latency               # latency, CUDA graphs, kernel profile

# the fixed-split experiments behind results/eval and the accuracy column of the latency table:
python kaggle/push.py finetune --mode full && python kaggle/push.py eval --mode full

# CPU only, from the detections:
python -m monodist.crossval --root data/kitti_tracking --runs results/cv/det --out results/cv
python -m monodist.evaluate --root data/kitti_tracking --det results/det/coco_1280 results/det/ft_1280 ...
python -m monodist.report --write
```

`results/cv/det/` holds the detections of the configurations kept in each fold and `results/det/` the COCO
detections and the fixed-split experiments, so the evaluation reruns on a CPU. Rerunning the choice of
configuration needs the detections of every fold and configuration, attached with the fold weights to the
GitHub release `v0.2` (unzip into `results/cv/det/`). `results/MANIFEST.json` lists the commit, software and
hardware behind each result.

---

## Layout

| File | Role |
|---|---|
| `monodist/kitti.py` | labels, calibration, the fixed split and the three folds |
| `monodist/geometry.py` | the two formulas, the MLP features, IoU |
| `monodist/match.py` | matching with KITTI's ignore rules, AP, F1 threshold, recall by distance |
| `monodist/mlp.py`, `metrics.py` | the distance MLP; AbsRel and the cluster bootstrap |
| `monodist/detect.py`, `finetune.py`, `yolo_data.py` | YOLO26n inference, fine-tuning configurations, dataset conversion |
| `monodist/latency.py`, `profile_gpu.py` | per-stage timing; kernel count and CUDA-graph replay |
| `monodist/evaluate.py`, `crossval.py`, `report.py` | one split, the cross-validation, this README |
| `kaggle/` | the Kaggle kernels, the push script and the job queue |

---

## Data and license

The code is released under the **GNU AGPL-3.0** (see `LICENSE`), the license of Ultralytics, which it
depends on.

The labels and calibration in `data/kitti_tracking/` and every result derived from KITTI come from the
**KITTI Vision Benchmark Suite** (A. Geiger, P. Lenz, R. Urtasun, "Are we ready for Autonomous Driving?
The KITTI Vision Benchmark Suite", CVPR 2012), licensed under **CC BY-NC-SA 3.0**: non-commercial use
only, with attribution, under the same license. See `data/kitti_tracking/README.md`.
