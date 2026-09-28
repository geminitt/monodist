# monodist

Detect cars and pedestrians in KITTI video with YOLO26n, estimate how far away they are from a single
camera, and measure what each stage of the pipeline costs on a Kaggle T4.

<!-- README.md is generated from README.template.md and results/ by `python -m monodist.report --write`;
     CI fails when they disagree. Edit the template, never README.md. -->

## Result in one paragraph

Over a three-fold cross-validation that tests every one of the 21 labelled KITTI sequences once, a
5k-parameter MLP that reads only the detected box estimates distance with a **mean relative error of
{coco.mlp.ci}** (median {coco.mlp.median}), using YOLO26n with its COCO weights and no detector training at
all. The textbook pinhole formula Z = f H / h gets {coco.size.mean} and the flat-road formula
{coco.ground.mean}. Fine-tuning the detector on KITTI raises mAP50-95 from {coco.map} ± {coco.map_sd} to
**{finetuned_nms.map} ± {finetuned_nms.map_sd}**, but it does not make the distances better
({finetuned_nms.mlp.ci}; the same {finetuned_nms.mlp.track} as COCO when every object counts once). On the
T4, decoding the PNG costs more than running the network; TensorRT fp16 cuts inference from
{lat.ft_1280_nms.inference} to {lat.ft_1280_nms_trt16.inference} ms at the same accuracy, and decoding the
next frame on a worker thread takes the fine-tuned detector with NMS to **{lat.ft_1280_nms_trt16_pipelined.fps}
frames per second**.

## What is compared

| Axis | Options |
|---|---|
| Detector | YOLO26n with its COCO weights (never trained); YOLO26n fine-tuned on KITTI, per fold |
| Fine-tuning configuration, chosen per fold on val | a: defaults · b: lower learning rate · c: Van and Cyclist labelled as their own classes · d: c with the DontCare regions painted grey |
| YOLO26 head | one-to-many head followed by NMS; NMS-free (one-to-one) head |
| Distance from a box | **known size** Z = f H / h (H = mean class height); **ground plane** Z = f h_cam / (v2 - c_y) (bottom edge on a flat road); **MLP** trained on the detector's own boxes |
| Speed levers | fp16, TensorRT fp16, CUDA graphs, input 1280 vs 640, decoding the next frame on a worker thread |

## Data and protocol

- **KITTI tracking**, the 21 labelled sequences (8,008 frames), classes Car and Pedestrian; the official test
  sequences have no public labels. Ground truth is the depth z of the LiDAR-based 3D box in camera 2, with
  the calibration of each sequence (four calibrations, f = 707 to 722 px).
- **Three-fold cross-validation by sequence** (`kitti.FOLDS`): each sequence is test exactly once; the other
  sequences of a fold split about 70/30 into train and val. Folds balance cars (instances and tracks) and
  pedestrian tracks; pedestrian instances cannot be balanced, since sequence 0019 alone holds 53% of them.
- **Per fold**: configurations a-d are fine-tuned on train (epoch chosen by Ultralytics on val; for c and d
  its val score also averages the Van and Cyclist classes) and the one with the best val mAP50-95 on Car and
  Pedestrian, scored with the project's own matcher, is kept ({chosen}). The known-size
  heights and the camera height come from the train labels; the MLP trains on the detector's boxes on the
  train sequences (epochs by leave-one-sequence-out, five seeds averaged); the score threshold of each class
  is the F1-best one on val. Nothing is chosen on test.
- **Matching**: greedy by score, IoU >= 0.5, per class; as in the KITTI devkit a car detection on a Van and a
  pedestrian detection on a sitting Person count as neither hit nor false alarm, and detections inside
  DontCare regions are ignored. mAP is COCO-style (101-point, IoU 0.50:0.95), per fold, reported as mean ±
  standard deviation over the three folds.
- **Distance metric**: AbsRel = |Zhat - Z| / Z on matched test objects farther than 2 m, pooled over the
  folds. Intervals are **cluster bootstrap over tracks**: a car seen in 300 frames is one sample, not 300.
  Next to the usual per-frame mean, the per-object mean gives every tracked object the same weight.

## Results

All tables, generated from `results/`, are also in [results/summary.md](results/summary.md).

### Choice of the fine-tuning configuration (val mAP50-95 per fold; bold = kept)

{table:selection}

Labelling vans and cyclists, with or without greying DontCare regions, never won a fold (c came second in
fold 1); a lower learning rate won two folds by about a point over the defaults.

### Detection (test, per fold, mean ± standard deviation over folds)

{table:detection}

### Distance error on the test objects of all folds

{table:distance}

Test set: {coco.n} object-frames from {coco.tracks} tracks for the COCO detector.

### Recall by distance at the operating point (IoU >= 0.5)

{table:recall}

### Latency

One Kaggle session (T4, 4 vCPU Xeon), one frame at a time, 300 frames of sequence 0007 x 3 rounds after 30
warm-up frames; medians in ms. "Pipelined" decodes the next frame on a worker thread while the GPU works on
the current one: its FPS is throughput, the latency of one frame stays about the serial total. FPS is the
mean of the three rounds (the session's first configuration ran its first round at 24 FPS, a cold start;
every other round is within 5 FPS of its configuration's mean). The accuracy
column is mAP50-95 of the same weights on the earlier fixed split (see `results/eval/`).

| Configuration | mAP50-95 | decode | preprocess | inference | postprocess | total, serial | FPS serial | FPS pipelined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO weights, PyTorch fp32, 1280 | {fixed.coco_1280.map} | {lat.coco_1280.decode} | {lat.coco_1280.preprocess} | {lat.coco_1280.inference} | {lat.coco_1280.postprocess} | {lat.coco_1280.total} | {lat.coco_1280.fps} | |
| fine-tuned, PyTorch fp32, 1280 | {fixed.ft_1280.map} | {lat.ft_1280.decode} | {lat.ft_1280.preprocess} | {lat.ft_1280.inference} | {lat.ft_1280.postprocess} | {lat.ft_1280.total} | {lat.ft_1280.fps} | {lat.ft_1280_pipelined.fps} |
| fine-tuned, PyTorch fp16, 1280 | {fixed.ft_1280_fp16.map} | {lat.ft_1280_fp16.decode} | {lat.ft_1280_fp16.preprocess} | {lat.ft_1280_fp16.inference} | {lat.ft_1280_fp16.postprocess} | {lat.ft_1280_fp16.total} | {lat.ft_1280_fp16.fps} | |
| fine-tuned, torch.compile (CUDA graphs), 1280 | | {lat.ft_1280_cg.decode} | {lat.ft_1280_cg.preprocess} | {lat.ft_1280_cg.inference} | {lat.ft_1280_cg.postprocess} | {lat.ft_1280_cg.total} | {lat.ft_1280_cg.fps} | |
| fine-tuned + NMS, PyTorch fp32, 1280 | {fixed.ft_1280_nms.map} | {lat.ft_1280_nms.decode} | {lat.ft_1280_nms.preprocess} | {lat.ft_1280_nms.inference} | {lat.ft_1280_nms.postprocess} | {lat.ft_1280_nms.total} | {lat.ft_1280_nms.fps} | {lat.ft_1280_nms_pipelined.fps} |
| fine-tuned, TensorRT fp16, 1280 | {fixed.ft_1280_trt16.map} | {lat.ft_1280_trt16.decode} | {lat.ft_1280_trt16.preprocess} | {lat.ft_1280_trt16.inference} | {lat.ft_1280_trt16.postprocess} | {lat.ft_1280_trt16.total} | {lat.ft_1280_trt16.fps} | {lat.ft_1280_trt16_pipelined.fps} |
| **fine-tuned + NMS, TensorRT fp16, 1280** | **{fixed.ft_1280_nms_trt16.map}** | {lat.ft_1280_nms_trt16.decode} | {lat.ft_1280_nms_trt16.preprocess} | {lat.ft_1280_nms_trt16.inference} | {lat.ft_1280_nms_trt16.postprocess} | {lat.ft_1280_nms_trt16.total} | {lat.ft_1280_nms_trt16.fps} | **{lat.ft_1280_nms_trt16_pipelined.fps}** |
| fine-tuned, PyTorch fp32, 640 | {fixed.ft_640.map} | {lat.ft_640.decode} | {lat.ft_640.preprocess} | {lat.ft_640.inference} | {lat.ft_640.postprocess} | {lat.ft_640.total} | {lat.ft_640.fps} | |
| fine-tuned, TensorRT fp16, 640 | {fixed.ft_640_trt16.map} | {lat.ft_640_trt16.decode} | {lat.ft_640_trt16.preprocess} | {lat.ft_640_trt16.inference} | {lat.ft_640_trt16.postprocess} | {lat.ft_640_trt16.total} | {lat.ft_640_trt16.fps} | {lat.ft_640_trt16_pipelined.fps} |

## Findings

1. **The box already contains most of the distance.** The MLP beats both formulas for both classes: for
   cars at 10-20 m it scores {coco.mlp.Car.10-20} against {coco.size.Car.10-20} for the known-size formula,
   and for cars closer than 10 m, most of them cut by the image border, {coco.mlp.Car.0-10} against
   {coco.size.Car.0-10}.
2. **The known-size formula has a floor of about 10% on cars, even with perfect boxes.** KITTI measures depth
   to the centre of the car, but a 2D box encloses the whole car: its bottom edge comes from the nearest face
   and its top edge from the far end of the roof. For a car seen from behind, with the ground y metres below
   the camera, Zhat / z is about 1 / (1 + (l / 2z)(2y - H) / H), about 10% low at 25 m for a 4.3 m car,
   which matches the labels (study notes, notebook 01). On the labelled boxes the formula scores
   {coco.floor.size.Car.whole} on untruncated cars. A detector whose boxes look *more* like the labels makes
   it *worse*: {finetuned_nms.size.Car.whole} after fine-tuning against {coco.size.Car.whole} with the COCO
   weights.
3. **The flat-road formula does not hold on KITTI.** The labelled ground points of cars lie between 1.38 and
   1.80 m below the camera (interquartile range), because of slopes and pitch; pedestrians stand about 16 cm
   higher, on the pavement; and half a degree of pitch moves the horizon by 6 px, a 23% error at 40 m.
4. **Fine-tuning helps detection, not distance.** On the {compare.finetuned_nms.n} test object-frames both
   detectors found, the fine-tuned MLP differs by {compare.finetuned_nms.mlp} points of mean AbsRel; one car
   of sequence 0015 that stays 2-5 m from the camera, cut by the border, accounts for most of it, and the
   medians ({coco.mlp.median}, {finetuned_nms.mlp.median}) and per-object means ({coco.mlp.track},
   {finetuned_nms.mlp.track}) agree.
5. **After fine-tuning, YOLO26's NMS-free head lags behind its NMS head** ({finetuned_e2e.map} vs
   {finetuned_nms.map}), while with the COCO weights the two are equal on the fixed split
   ({fixed.coco_1280.map} vs {fixed.coco_1280_nms.map}). *Hypothesis tested and rejected:* the lag is not
   caused by choosing the epoch with the NMS head. Configuration e chose the epoch with the NMS-free head;
   on fold 0 it picked the same epoch ({hyp.e.epoch}), and the NMS-free head stays behind at every epoch of
   training (test {hyp.e.test_e2e} vs {hyp.e.test_nms}).
6. **Eager PyTorch is partly limited by kernel launches.** One forward pass launches
   {profile.nms.fp32.kernels_per_forward} CUDA kernels in fp32 ({profile.nms.fp16.kernels_per_forward} in
   fp16). Replaying them as one CUDA graph cuts the forward pass from {profile.nms.fp32.eager_wall_ms} to
   {profile.nms.fp32.graph_wall_ms} ms in fp32 and from {profile.nms.fp16.eager_wall_ms} to
   {profile.nms.fp16.graph_wall_ms} ms in fp16: the time eager PyTorch loses between kernels is what hides
   fp16's speed. Ultralytics' `compile="reduce-overhead"` did not capture that gain
   ({lat.ft_1280_cg.inference} ms), so the practical lever remains TensorRT.
7. **Far objects are the weak point.** Beyond 40 m the COCO model finds {coco.recall.Car.40} of the cars and
   {coco.recall.Pedestrian.40} of the {coco.recall.Pedestrian.total40} pedestrian object-frames; input 640
   instead of 1280 loses more far objects (fixed split).

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

## Limitations

- One fine-tuning run per fold and configuration; the spread over folds mixes data and seed variation.
- Pedestrians beyond 40 m are almost absent from KITTI tracking ({coco.recall.Pedestrian.total40}
  object-frames), and no detector finds them.
- The mean AbsRel is sensitive to a few objects seen for hundreds of frames; the per-object mean and the
  median are reported next to it.
- Latency is for one frame at a time on a Kaggle T4 with 4 CPU cores; PNG decoding depends on the CPU. The
  latency session used the fixed-split fine-tuned weights (same architecture as the fold models).
- GPU code paths (detection, fine-tuning, latency) are tested by running them on Kaggle, not by CI.

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
GitHub release `v0.2` (unzip into `results/cv/det/`).

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

## Data and license

The code is released under the **GNU AGPL-3.0** (see `LICENSE`), the license of Ultralytics, which it
depends on.

The labels and calibration in `data/kitti_tracking/` and every result derived from KITTI come from the
**KITTI Vision Benchmark Suite** (A. Geiger, P. Lenz, R. Urtasun, "Are we ready for Autonomous Driving?
The KITTI Vision Benchmark Suite", CVPR 2012), licensed under **CC BY-NC-SA 3.0**: non-commercial use
only, with attribution, under the same license. See `data/kitti_tracking/README.md`.
