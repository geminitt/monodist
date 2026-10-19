<div align="center">

# monodist

[![CI](https://img.shields.io/github/actions/workflow/status/geminitt/monodist/ci.yml?branch=main&label=CI&style=for-the-badge)](https://github.com/geminitt/monodist/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/PYTHON-3.12-A19654?style=for-the-badge)](./pixi.toml)
[![License](https://img.shields.io/badge/LICENSE-AGPL--3.0-6B7F4E?style=for-the-badge)](./LICENSE)

**Detect obstacles in KITTI video, estimate distance from one camera, track their closing motion,
and measure the complete pipeline on a T4.**

</div>

<!-- Generated from this template and results/rebuild/ by python -m monodist.report --write. -->

---

## Result

The fixed fine-tuned NMS pipeline reaches **{test.nms.map} five-class AP50–95** on the held-out sequences.
Its saved distance ensemble has **{test.nms.distance} mean depth AbsRel** on correctly matched detections
at least 2 m away. A causal Kalman filter reduces speed MAE from **{test.nms.speed.raw} to
{test.nms.speed.filtered} m/s** on matched observations with track history and a valid reference speed.
These populations differ; missed objects and false-positive distances are not included in the distance error.

On a Tesla T4, the same saved ensemble, thresholds, tracker and TTC path takes **{lat.nms.serial} ms/frame**
in serial processing. Overlapping the next PNG decode with current-frame processing gives **{lat.nms.fps} FPS**.
The latter is throughput, not capture-to-output latency.

---

## System

`image → YOLO26n boxes/classes → saved distance ensemble → depth and range → IoU tracker → Kalman filter → TTC`

| Component | Contract |
|---|---|
| Detector | YOLO26n; COCO baseline or full KITTI fine-tune; NMS and NMS-free heads |
| Classes | Car, Van, LargeVehicle (Truck/Tram), Pedestrian (Pedestrian/Person), Cyclist |
| Reporting groups | Vehicle, Person, Cyclist; also class-agnostic detection |
| Distance | Five MLPs trained from scratch; 11 box/class features, hidden layers 64/64, averaged in log depth |
| Geometry baselines | Known size, $Z = fH/h$; ground plane, $Z = f h_\mathrm{cam}/(v_2-c_y)$, with known-size fallback |
| Temporal output | Automatic same-class IoU association; causal constant-velocity Kalman filter on depth and depth rate |
| TTC | Optical-axis depth divided by closing speed; infinite when not approaching |

Depth is camera-2 optical-axis Z. Range is Euclidean distance to the labeled 3D box center; predictions use
estimated Z along the detected box-center ray. Reset tracker state between sequences. KITTI frames are 0.1 s apart.

---

## Data and evaluation protocol

- **KITTI tracking labeled sequences**, partitioned by complete sequence. Development uses 16 sequences;
  final TEST is **0003, 0007, 0011, 0016, 0017**. This is a local split of labeled data, not a submission to the
  official hidden-label KITTI tracking benchmark.
- **Three development folds** select detector configuration using validation AP. Final configuration is
  **{chosen}**, selected by mean validation AP across folds. The NMS-free comparison uses the same
  selected detector checkpoint. The final fine-tune uses the predefined train/validation split.
- **Distance fitting** uses detector boxes on training sequences. MLP epochs are chosen by grouped inner
  holdouts of those training sequences. Per-class confidence maximizes validation F1. Kalman settings use
  development validation predictions. Final TEST uses the exact saved ensembles/settings measured on T4.
  No final TEST metric selects a configuration or fits a model. Software tests may use TEST label fixtures.
- **Detection matching** is score-ordered greedy matching; IoU thresholds 0.50 through 0.95, 101-point AP,
  and Misc/DontCare ignore regions. This custom protocol is not the complete official KITTI difficulty protocol.
- **Distance** is scored on correctly classified matches at IoU ≥ 0.5 and Z ≥ 2 m. Read the sample counts
  and recall tables alongside error. Mean depth and range AbsRel use track-cluster bootstrap 95% intervals
  with 2,000 resamples. The median is descriptive. Tracks are identified jointly by sequence and object id.
- **Development AP** is mean ± sample standard deviation across folds; distance rows are pooled without
  duplicate object/frame keys. TEST AP is one fixed-split result. Comparisons between detectors use common
  matched object/frame keys and do not measure overall detection coverage.
- **Temporal scoring** matches kept detections class-agnostically at IoU ≥ 0.5. All kept predictions,
  including false positives and ignored detections, enter tracking before labels are attached for scoring.
  Raw and filtered speed use the same eligible matched rows with history. Reference speed is a centered
  five-frame label slope; future labels enter only that reference. TTC scoring uses reference TTC below 10 s.

### Development selection

Configuration a uses defaults; b lowers the learning rate. Every configuration uses the same five classes.

{table:selection}

---

## Detection

COCO has no direct Van/Cyclist predictions in this mapping; five-class AP includes those missed classes.
Grouped and class-agnostic AP therefore answer different questions from five-class AP.

{table:detection}

## Distance

Each detector has its own population of correctly matched objects. The pooled means alone are not a paired
comparison between detectors. Range and depth have different geometric targets.

{table:distance}

### Final TEST: MLP by class

A dash means there were no eligible matches, not zero error.

{table:classes}

### Paired detector comparison

Difference is fine-tuned minus COCO mean depth AbsRel on common matches. Positive values favor COCO.
Intervals resample complete ground-truth tracks; an interval crossing zero does not establish a mean difference.

{table:paired}

### Final TEST recall by true depth

Cells show recall and the number of labeled object-frames. Recall includes missed objects, which the
matched-distance table excludes.

{table:recall}

## Tracking, speed and TTC

The filter improves the measured speed/TTC outcomes, while persistent per-object distance bias remains.
The identity-continuity diagnostics in the recorded outputs are not official MOT/HOTA/MOTA scores.

{table:temporal}

## T4 timing

One session, sequential configurations on GPU 0, with four CPU-affinity slots; sequence 0019, 300 PNGs,
three rounds after 30 warm-up images, batch 1, input 1280, FP32 PyTorch. All configurations use the same T4
UUID, environment and image hashes. The second provisioned T4 is idle under the approved controlled-timing
protocol. Final detector configuration and saved distance-model hashes match the TEST quality evaluation.

Stage values are per-frame medians. Total also includes file reads, preprocessing, postprocessing and Python
overhead, so it is not the sum of the displayed subset. FPS is the median of three round wall-clock rates,
including round setup/drain. Repeated images can benefit from storage caching. Decode-overlapped FPS measures
processing cadence; it does not establish per-image capture latency or arbitrary-camera real-time behavior.

{table:latency}

---

## Relation to published work

| Reference | Connection | Protocol difference |
|---|---|---|
| [MonoLoco, Bertoni et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Bertoni_MonoLoco_Monocular_3D_Pedestrian_Localization_and_Uncertainty_Estimation_ICCV_2019_paper.html) | Monocular pedestrian localization and uncertainty from 2D human pose | This project uses boxes/classes, five obstacle classes and a tracking-sequence split; its AbsRel is not a comparable leaderboard score |
| [KITTI tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) | Driving sequences and labeled object identities | This project reports custom detection, distance and optical-axis temporal metrics; it is not an official tracking submission |
| [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) | Detector family with NMS-free inference support | Published detector timings exclude this project's saved ensemble, image decoding, tracker and TTC; compare the same workload before comparing speed |

No numerical superiority over these systems is claimed under unmatched protocols.

## Limits

- One detector training run per configuration/fold; fold spread does not isolate seed variation.
- Distance errors exclude misses and false-positive distances. Van/Cyclist coverage differs strongly from COCO.
- Long tracks influence frame-weighted means. Track resampling does not model every dependence across
  objects or across the fold training runs.
- Distance fitting sees training-image detector boxes; their distribution can differ from unseen-image boxes.
  The effect on generalization was not isolated. Near/truncated objects remain a weakness in development.
- Camera features use one focal length; the ground-plane baseline assumes a level camera and flat road.
  Arbitrary camera geometry and road pitch are not verified.
- TTC describes optical-axis closing motion, not whether two trajectories will collide. The KITTI metrics
  and T4 timings do not evaluate the separate illustrative simulation controller or establish driving safety.
- Training with RAM caching emitted nondeterminism warnings. Source/input hashes do not imply bit-identical
  retraining. TensorRT, FP16 and compiled variants are not part of the rebuilt timing matrix.

---

## Local closed-loop demo and portable delivery

### Real-video playback

`/video` processes a calibrated KITTI 0012 clip on demand: actual video decode, fixed YOLO/distance model,
tracker/TTC, annotated JPEG and browser playback. Play/pause/reset, frame stepping and native-rate/max-speed
playback are available. The page reports native FPS, server processing capacity, request-to-image decode
latency and observed playback FPS (including browser paint opportunities). No predicted frames are cached.

Include a calibrated **10 FPS** clip with `bundle build --video /path/to/clip.mp4`, or pass
`demo --video /path/to/clip.mp4` at launch. Video dimensions must match the bundle camera; optional one-pixel
encoder padding is removed before inference. The browser route is **http://127.0.0.1:8765/video**.
The default device is CPU. For a configured local CUDA environment, use:

```bash
pixi run -e gpu python -m monodist.demo --bundle /path/to/monodist-demo --video /path/to/clip.mp4 --device 0
```

Realtime means keeping pace with the source's 10 FPS; this does not establish 30 FPS operation. Actual FPS
depends on device, current load, decoding and browser transport. This recorded-video mode cannot change the
future scene in response to braking. True 3D control-loop work is separate from this playback check.

### 2.5D feasibility prototype

The browser demo renders a 2.5D photographic camera, runs **actual YOLO26n → saved distance ensemble →
tracker/Kalman → stopping-distance policy**, and applies acceleration to the next world state. Ground truth
enters rendering and diagnostics only. No detections or distance predictions are replaced with world truth.

Build a local archive from the fixed fine-tuned NMS detector and its matching timed distance checkpoint:

```bash
pixi run python -m monodist.bundle build --weights /path/to/best.pt --distance /path/to/distance.pt --images data/kitti_tracking/training/image_02/0012 --calibration data/kitti_tracking/training/calib/0012.txt --frames 78 --source-commit {source} --out /path/to/monodist-demo.zip
```

The archive includes unchanged checkpoints, saved confidence/filter settings, calibration, all 78 development
sample images, source, locked dependencies and `START_HERE.md`. Extract it, enter `monodist-demo`, then run:

```bash
pixi install --frozen
pixi run python -m monodist.bundle verify --bundle .
pixi run python -m monodist.bundle run --bundle . --frames 6
pixi run python -m monodist.demo --bundle .
```

Open **http://127.0.0.1:8765**. Use Run/Pause, Step, Reset, scenario selection and automatic-braking on/off.
Dependencies require an initial download; models are local and inference runs offline. Linux/WSL and pixi
are required. The server binds to loopback and shares one episode across tabs. Stop it with Ctrl+C.

A local actual-model check on the rendered scenes gave these outcomes. They illustrate this renderer/policy,
not real-world driving performance or an additional KITTI benchmark:

| Scenario | Automatic braking on | Automatic braking off |
|---|---|---|
| Stationary vehicle | Stopped without collision, 5.36 m center-depth remaining | Collision |
| Lead vehicle brakes | Collision at 3.53 m/s ego speed | Collision at 15 m/s |
| Vehicle outside corridor | No braking, no collision | No collision |
| Late obstacle | Collision at 12.92 m/s ego speed | Collision at 15 m/s |

The car is a scaled KITTI photograph on a fixed background. Geometry, motion, lighting and occlusion are
simplified. The distance estimates are biased on these rendered images, and detections can disappear.
The UI exposes those outputs and retains failure cases. The policy uses illustrative 6 m/s² braking;
it is not trained or validated for driving. Simulation time advances by 0.1 s per step independently of
inference wall time. This demo does not replace the frozen quality/latency measurements above.

---

## Provenance and reproduction

Measurement library source is fixed at **`{source}`**. Training version 4, development analysis and timing
version 8 use that source. Final scoring calls those frozen library functions with the saved timed models;
its orchestration driver is recorded separately by SHA-256. Report-rendering changes do not refit models.

[results/rebuild/MANIFEST.json](results/rebuild/MANIFEST.json) identifies source/input/model hashes,
environments, hardware, run versions, fixed choices and packaged files. The archived scoring/packaging drivers
record their original workspace paths; adapt the directory constants and use fresh output folders to reproduce
elsewhere. Large weights, detections and per-frame intermediates remain in the original Kaggle outputs and
local runs; this update does not claim a new public artifact release. A fresh clone can verify the report,
but full model reproduction needs those artifacts or a new training/detection run.

```bash
pixi run test
pixi run python -m monodist.report --check
pixi run python -m monodist.report --summary

# Build the pinned kernels without spending GPU quota:
python kaggle/push.py train --mode full --commit {source} --build-only
python kaggle/push.py eval --mode full --commit {source} --set DEADLINE=3600 --build-only
```

The generated tables use only [results/rebuild/summary.json](results/rebuild/summary.json).
Older result directories are retained for provenance and are not inputs to the rebuilt README.

## Layout

| Path | Role |
|---|---|
| `monodist/kitti.py`, `geometry.py`, `match.py` | Data, camera geometry, class mapping and scoring |
| `monodist/mlp.py`, `evaluate.py`, `crossval.py` | Distance models, one-split evaluation and development folds |
| `monodist/pipeline.py`, `temporal.py` | Shared saved-model inference, association, filtering and TTC |
| `monodist/analysis.py`, `calibrate.py` | Signed development analysis and final development-only calibration |
| `monodist/detect.py`, `finetune.py`, `latency.py` | GPU detections, training and saved-pipeline timing |
| `monodist/video.py` | Consecutive calibrated video inference and processing-timing checks |
| `monodist/bundle.py`, `demo.py`, `simulation.py`, `web/dist/` | Portable fixed-model archive, local browser UI and learned-perception control loop |
| `monodist/report.py`, `README.template.md` | Generate this document from recorded measurements |
| `kaggle/` | Pinned remote jobs and orchestration |

---

## Data and license

Code: **GNU AGPL-3.0**, see [LICENSE](LICENSE). KITTI labels, calibration and derived results:
**CC BY-NC-SA 3.0**, with attribution to Geiger, Lenz and Urtasun, *Are we ready for Autonomous Driving?
The KITTI Vision Benchmark Suite*, CVPR 2012. See [data attribution](data/kitti_tracking/README.md).
