# monodist

Detect cars and pedestrians in KITTI video with YOLO26n and estimate how far away they are from a single
camera, then measure what each stage of the pipeline costs on a Kaggle T4.

## Result in one paragraph

With the COCO-pretrained YOLO26n and no training at all, a 5k-parameter MLP that reads only the detected
box (its size and position, divided by the focal length) estimates distance with a **mean relative error of
7.9%** [95% CI 6.7-9.3] on the test sequences, **1.5 m** on average. The textbook pinhole formula
Z = f H / h gets 18.1% (7.9% on objects not cut by the image border) and the flat-road formula 31.5%.
Fine-tuning the detector on KITTI raises test mAP50-95 from 0.413 to **0.477**, but it does **not** make
the distances better: the gap in the headline mean comes from one car parked next to the camera. On the
T4, decoding the PNG takes longer than running the network; TensorRT fp16 cuts inference from 10.5 to 3.9 ms
at the same accuracy, and decoding the next frame on a worker thread brings the fine-tuned detector with NMS
to **71 frames per second** (mAP50-95 0.476).

## What is compared

| Axis | Options |
|---|---|
| Detector | YOLO26n with its COCO weights (no training); YOLO26n fine-tuned on the KITTI train sequences |
| YOLO26 head | NMS-free (one-to-one) head; one-to-many head followed by NMS |
| Distance from a box | **known size** Z = f H / h (H = mean class height); **ground plane** Z = f h_cam / (v2 - c_y) (bottom edge on a flat road); **MLP** trained on the detector's own boxes |
| Speed levers | fp16, TensorRT fp16, input 1280 vs 640, decoding the next frame on a worker thread |

## Data and protocol

- **KITTI tracking**, the 21 labelled sequences (8,008 frames), classes Car and Pedestrian. The official test
  sequences have no public labels, so the 21 are split **by sequence** (never by frame):
  train 0001 0005 0008 0009 0013 0018 0019 0020 / val 0003 0004 0006 0011 0016 /
  test 0000 0002 0007 0010 0012 0014 0015 0017, about 60/20/20 of the objects and tracks of both classes.
- **Ground-truth distance** is the depth z of the labelled 3D box in camera 2 (LiDAR-based labels). The four
  calibrations of the dataset (f = 707 to 722 px) are used per sequence.
- **Matching**: greedy by score, IoU >= 0.5, class by class. As in the KITTI devkit, a car detection on a Van
  and a pedestrian detection on a sitting Person count as neither hit nor false alarm, and detections inside
  DontCare regions are ignored. mAP is COCO-style (101-point, IoU 0.50:0.95).
- **Operating point**: per detector and class, the score threshold with the best F1 on the val sequences.
- **Distance metric**: AbsRel = |Zhat - Z| / Z on the matched test objects farther than 2 m (closer ones are
  all cut by the image border). Intervals are **cluster bootstrap over tracks**: one car seen in 300 frames
  is one sample, not 300.
- **Constants and training**: the class heights (car 1.52 m, pedestrian 1.74 m) and the camera height
  (1.51 m) come from the train labels; the MLP trains on the detector's boxes on the train sequences (epochs
  picked by leave-one-sequence-out, final prediction averaged over 5 seeds); the fine-tuned detector's
  epoch is picked on val.

## Results

The full tables, generated from `results/`, are in [results/summary.md](results/summary.md).

### Detection on the test sequences

| Detector | Head | Car AP50-95 | Pedestrian AP50-95 | mAP50-95 | mAP50 |
|---|---|---:|---:|---:|---:|
| COCO weights | NMS-free | 0.533 | 0.294 | 0.413 | 0.746 |
| COCO weights | NMS | 0.534 | 0.293 | 0.414 | 0.752 |
| fine-tuned | NMS-free | 0.622 | 0.280 | 0.451 | 0.736 |
| fine-tuned | NMS | **0.650** | **0.303** | **0.477** | **0.784** |

### Distance error, mean AbsRel [95% CI] (median)

| Detector | Method | All objects | Cars not truncated | Pedestrians not truncated |
|---|---|---|---|---|
| COCO weights | known size | 18.1% [11.2, 29.2] (7.3%) | 7.9% [7.1, 8.9] (6.8%) | 8.1% [6.7, 9.6] (5.8%) |
| COCO weights | ground plane | 31.5% [24.8, 39.0] (18.9%) | 26.9% [21.4, 34.1] (18.7%) | 31.5% [17.3, 45.6] (12.9%) |
| COCO weights | **MLP** | **7.9% [6.7, 9.3] (5.2%)** | **5.9% [5.1, 6.8] (4.7%)** | **6.0% [4.8, 7.3] (4.3%)** |
| fine-tuned, NMS | known size | 17.1% [11.7, 26.1] (9.0%) | 9.8% [8.8, 10.8] (8.5%) | 8.2% [6.2, 10.7] (6.3%) |
| fine-tuned, NMS | ground plane | 31.8% [24.9, 39.6] (18.9%) | 28.9% [22.0, 37.5] (18.9%) | 25.4% [15.1, 35.3] (12.1%) |
| fine-tuned, NMS | MLP | 13.5% [7.6, 24.6] (5.5%) | 6.5% [5.7, 7.4] (5.0%) | 7.1% [5.1, 9.3] (4.6%) |

Test set: about 6,000 matched object-frames from 140 tracks.

### Latency

All timings come from one Kaggle session (T4, 4 vCPU Xeon 2.0 GHz), one frame at a time, 300 frames of test
sequence 0007 x 3 rounds after 30 warm-up frames; medians in ms. "Serial" runs the stages one after the other;
"pipelined" decodes the next frame on a worker thread while the GPU works on the current one.

| Configuration | mAP50-95 | decode | preprocess | inference | postprocess | total, serial | FPS serial | FPS pipelined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO weights, PyTorch fp32, 1280 | 0.413 | 13.2 | 3.5 | 10.7 | 0.9 | 30.8 | 32 | |
| fine-tuned, PyTorch fp32, 1280 | 0.451 | 13.1 | 3.9 | 10.5 | 0.8 | 30.7 | 32 | 65 |
| fine-tuned, PyTorch fp16, 1280 | 0.451 | 12.2 | 2.7 | 11.3 | 0.5 | 29.1 | 34 | |
| fine-tuned + NMS, PyTorch fp32, 1280 | 0.477 | 12.5 | 3.6 | 9.7 | 1.5 | 29.2 | 34 | 59 |
| fine-tuned, TensorRT fp16, 1280 | 0.451 | 12.1 | 2.7 | 4.0 | 0.6 | 21.6 | 46 | 71 |
| **fine-tuned + NMS, TensorRT fp16, 1280** | **0.476** | 12.0 | 2.7 | 3.9 | 1.5 | 22.2 | 45 | **71** |
| fine-tuned, PyTorch fp32, 640 | 0.358 | 12.2 | 1.1 | 10.1 | 0.5 | 26.2 | 38 | |
| fine-tuned, TensorRT fp16, 640 | 0.358 | 12.0 | 1.1 | 2.7 | 0.6 | 18.6 | 53 | 68 |

Serial total also includes reading the file (~1 ms), Ultralytics' own overhead (~0.5 ms) and the distance
step (0.7 ms for the features, one MLP and both formulas; the five-seed ensemble that was evaluated runs four
more MLPs, about 0.1 ms on a laptop CPU). Pipelined "FPS" is throughput; the latency of one frame is still
about the serial total. FPS varies across the three rounds by up to 8 for the PyTorch NMS rows
and up to 4 elsewhere (per-round values in the summary). Full per-stage tables with bootstrap intervals are
in [results/summary.md](results/summary.md).

What the table says:

- **Decoding the PNG (12-13 ms) is the slowest stage**, slower than the network. Once the network is fast,
  the CPU decides the frame rate.
- **fp16 in PyTorch buys nothing** (11.3 vs 10.5 ms): at batch 1 a 2.4M-parameter network is bound by the
  cost of launching its many small GPU kernels, not by arithmetic. **TensorRT**, which fuses those kernels,
  runs it in **4.0 ms** (2.6x) with the same accuracy.
- **NMS costs about 1 ms** (postprocess 1.5 vs 0.6 ms) and, after fine-tuning, is worth 2.5 mAP points.
- **Input 640 saves only 1.3 ms of TensorRT inference** but loses 12 mAP points and most far objects.
- **Pipelining the decode** doubles PyTorch throughput (32 to 65 FPS) and takes TensorRT from 45 to 71 FPS;
  beyond that the decode thread itself is the limit (~80 FPS at 12.5 ms per frame).

## Findings

1. **The box already contains most of the distance.** With the COCO detector, an MLP on seven numbers per
   box beats both formulas in every class and distance bin except the five pedestrian frames beyond 40 m.
   It halves the error of the known-size formula on
   cars between 10 and 20 m (5.4% vs 9.5%) and handles truncated boxes far better (17.6% vs 73.7% for cars
   closer than 10 m, most of which are cut by the border).
2. **The known-size formula has a floor of about 9% on cars, even with perfect boxes.** KITTI measures depth
   to the centre of the car, but a 2D box encloses the whole car: its bottom edge comes from the nearest face
   and its top edge from the farthest part of the roof. For a car seen from behind, with the ground y metres
   below the camera, Zhat / z is about 1 / (1 + (l / 2z)(2y - H) / H): about 10% low at 25 m for a 4.3 m car,
   which matches the labels (study notes, notebook 01). On the labelled boxes themselves the formula scores
   8.9% on untruncated cars. A detector whose boxes look *more* like the labels makes this formula *worse*:
   9.8% after fine-tuning against 7.9% with the COCO weights.
3. **The flat-road formula does not hold on KITTI.** The labelled ground points of cars lie between 1.38 and
   1.80 m below the camera (interquartile range), because of slopes and pitch; pedestrians stand about 16 cm
   higher, on the pavement; and the bottom edge of a car box is its nearest corner, not its centre. Beyond
   20 m, where the bottom edge is only a few pixels below the horizon, these errors dominate.
4. **Fine-tuning helps detection, not distance.** It raises mAP50-95 by 6.4 points (with NMS), mostly through
   tighter car boxes, but its MLP is not better. On the 5,496 test object-frames that both detectors found,
   the fine-tuned MLP is 5.9 points worse on average [0.5, 15.9]. One car accounts for 5.0 of those 5.9
   points: in sequence 0015 it stays 2-5 m from the camera, cut by the image border, for 300 frames.
   Without it the two are within a point (7.2% vs 8.1%).
5. **After fine-tuning, YOLO26's NMS-free head lags behind its NMS head** (0.451 vs 0.477 mAP50-95), while
   with the COCO weights the two are equal (0.413 vs 0.414). The fine-tuned NMS-free head also stretches
   9.5% of its boxes to the bottom edge of the image, against 1% for the NMS head and 6% in the labels.
   The best epoch (the 4th) was chosen by Ultralytics' validation, which scores the NMS head, so nothing
   selected for the one-to-one head, and after four epochs it may simply be undertrained.
6. **Far objects are the weak point.** Beyond 40 m the COCO model finds 63% of the cars and 7% of the
   pedestrians; the fine-tuned model finds fewer far cars with its NMS-free head (42%) and none of the 76 far
   pedestrians. Input 640 instead of 1280 loses more: 29% of far cars.

## Checks

- **AP implementation**: on the val sequences, without the ignore rules and with NMS, it gives mAP50-95
  0.466 (Car 0.548, Pedestrian 0.384) against 0.465 (0.552, 0.378) from Ultralytics' own validation of the
  same weights.
- **Labels**: the 3D boxes projected with P2 overlap the 2D labels at a median IoU of 0.97.
- **Formulas**: both invert the pinhole projection exactly on synthetic boxes (unit tests); the flat-road
  formula applied to the projected bottom centre of real labels recovers their height to the millimetre.
- **Reproducibility**: the COCO detections from a GPU run and from a CPU run agree on 46,230 of 46,231 boxes
  (score >= 0.05) at IoU > 0.999; rerunning the evaluation reproduces its JSON byte for byte.
- **Precision**: PyTorch fp32, fp16 and TensorRT fp16 give the same mAP50-95 to three decimals.
- **Every number in this README** was recomputed from `results/` by a script before publishing.

## Limitations

- The fine-tuning labels contain only Car and Pedestrian; vans, cyclists and DontCare regions are left
  unlabelled in the training images, so the detector learns them as background. The evaluation ignores
  detections on vans and DontCare regions, as the KITTI devkit does, but a COCO "person" on a cyclist counts
  as a false alarm.
- The test split has 30 pedestrian tracks and 76 pedestrian object-frames beyond 40 m; pedestrian intervals
  are wide and far pedestrians are barely measured.
- One fine-tuning run, one split: the detector numbers carry no seed-to-seed uncertainty.
- Timings are for one frame at a time on a Kaggle T4 with 4 CPU cores; PNG decoding depends on the CPU.

## Mistakes caught while doing this

Each one was found by checking a number that looked too good or too odd, and fixed before the results above.

1. **The distance MLP first trained on the val sequences**, where 91% of the pedestrians come from one
   sequence; it was worse than the plain formula on pedestrians (22.9% vs 8.7%). It now trains on the train
   sequences.
2. **Both TensorRT engines were written to the same file**, so the "1280" timing measured the 640 engine;
   the engine detections also failed because Ultralytics stacks a list of images into one batch and the
   engines are fixed at batch 1. The latency tool now refuses to run an engine at a shape other than the one
   asked for.
3. **A fixed score threshold of 0.25** favoured whichever detector outputs higher scores; each detector now
   uses its F1-best threshold from val.
4. **One MLP seed** moved the error by about a point between two configurations with identical boxes; the
   prediction is now the average of five seeds.
5. **Objects closer than 2 m** (all truncated, some at 0.6 m) made AbsRel explode; they are excluded and
   counted.
6. **Latency from different Kaggle sessions** may come from different machines; the final timings are all
   from one session.

## Reproduce

```bash
pixi install && pixi run test          # 31 tests on CPU, about 30 s (labels needed in data/, see below)

# Kaggle (GPU T4, dataset leducnhuan/kitti-tracking), from the repo root, code pinned to the pushed HEAD:
python kaggle/push.py finetune --mode full      # ~40 min
python kaggle/push.py eval --mode full          # detections of every configuration + latency, ~1 h
kaggle kernels output spritker/monodist-eval -p runs/kaggle/eval

# Local, CPU only (~7 min for eight configurations):
python -m monodist.evaluate --root data/kitti_tracking --det results/det/coco_1280 results/det/ft_1280_nms ...
python -m monodist.report > results/summary.md
```

`results/det/` holds every detection of every configuration (about 70 MB), so the evaluation reruns on a CPU
without Kaggle. `data/kitti_tracking/training/{label_02,calib}` (9 MB) is enough for the tests and the evaluation;
`kaggle datasets download leducnhuan/kitti-tracking -f kitti_tracking/training/label_02/0000.txt` fetches
one file at a time.

## Layout

| File | Role |
|---|---|
| `monodist/kitti.py` | labels, calibration, the split |
| `monodist/geometry.py` | the two formulas, the MLP features, IoU |
| `monodist/match.py` | matching with KITTI's ignore rules, AP, F1 threshold, recall by distance |
| `monodist/mlp.py` | the distance MLP |
| `monodist/metrics.py` | AbsRel, cluster bootstrap |
| `monodist/detect.py`, `finetune.py`, `yolo_data.py` | YOLO26n inference, fine-tuning, dataset conversion |
| `monodist/latency.py` | per-stage timing |
| `monodist/evaluate.py`, `report.py` | everything above, into `results/` |
| `kaggle/` | the Kaggle kernels and the script that pushes them |
