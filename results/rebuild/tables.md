## selection

| Development fold | Defaults: validation AP | Lower LR: validation AP | Chosen |
|---|---|---|---|
| 0 | 0.429 | 0.450 | b |
| 1 | 0.336 | 0.318 | a |
| 2 | 0.227 | 0.231 | b |

## detection

| Partition | Detector | 5-class AP50–95 | AP50 | 3-group AP50–95 | Class-agnostic AP50–95 |
|---|---|---|---|---|---|
| development | COCO NMS | 0.206 ± 0.066 | 0.367 ± 0.070 | 0.251 ± 0.030 | 0.418 ± 0.062 |
| development | Fine-tuned NMS | 0.308 ± 0.061 | 0.498 ± 0.087 | 0.335 ± 0.031 | 0.506 ± 0.062 |
| development | Fine-tuned NMS-free | 0.271 ± 0.029 | 0.435 ± 0.039 | 0.307 ± 0.025 | 0.466 ± 0.051 |
| test | COCO NMS | 0.202 | 0.361 | 0.261 | 0.414 |
| test | Fine-tuned NMS | 0.293 | 0.478 | 0.412 | 0.521 |
| test | Fine-tuned NMS-free | 0.270 | 0.439 | 0.379 | 0.488 |

## distance

| Partition | Detector | Method | Mean depth AbsRel [95% CI] | Median depth AbsRel | Mean range AbsRel [95% CI] | Object-frames | Tracks |
|---|---|---|---|---|---|---|---|
| development | COCO NMS | Known size | 12.32% [10.08, 15.87] | 6.74% | 11.13% [9.34, 13.84] | 22814 | 622 |
| development | COCO NMS | Ground plane | 25.82% [22.92, 29.09] | 16.94% | 24.99% [22.28, 27.96] | 22814 | 622 |
| development | COCO NMS | MLP ensemble | 8.34% [7.76, 8.96] | 5.55% | 8.15% [7.61, 8.74] | 22814 | 622 |
| development | Fine-tuned NMS | Known size | 14.27% [12.84, 16.36] | 10.38% | 13.21% [12.13, 14.64] | 24267 | 695 |
| development | Fine-tuned NMS | Ground plane | 25.37% [22.51, 28.33] | 17.18% | 24.33% [21.69, 27.00] | 24267 | 695 |
| development | Fine-tuned NMS | MLP ensemble | 10.13% [8.88, 11.67] | 5.61% | 9.38% [8.41, 10.49] | 24267 | 695 |
| development | Fine-tuned NMS-free | Known size | 14.39% [12.77, 16.73] | 10.62% | 13.25% [12.03, 14.81] | 21186 | 684 |
| development | Fine-tuned NMS-free | Ground plane | 24.99% [22.11, 28.38] | 16.80% | 23.86% [21.27, 26.73] | 21186 | 684 |
| development | Fine-tuned NMS-free | MLP ensemble | 9.82% [8.40, 11.77] | 5.26% | 9.03% [7.97, 10.32] | 21186 | 684 |
| test | COCO NMS | Known size | 13.30% [12.05, 14.71] | 9.12% | 12.31% [11.08, 13.62] | 7292 | 151 |
| test | COCO NMS | Ground plane | 21.94% [18.43, 25.96] | 17.34% | 21.25% [17.70, 25.29] | 7292 | 151 |
| test | COCO NMS | MLP ensemble | 9.24% [8.25, 10.48] | 6.66% | 9.04% [8.03, 10.28] | 7292 | 151 |
| test | Fine-tuned NMS | Known size | 14.89% [13.95, 15.90] | 12.41% | 14.03% [13.11, 14.93] | 7869 | 161 |
| test | Fine-tuned NMS | Ground plane | 20.68% [18.58, 23.02] | 17.76% | 19.84% [17.71, 22.13] | 7869 | 161 |
| test | Fine-tuned NMS | MLP ensemble | 9.13% [8.20, 10.26] | 6.07% | 8.93% [8.01, 10.07] | 7869 | 161 |
| test | Fine-tuned NMS-free | Known size | 14.64% [13.70, 15.62] | 12.00% | 13.82% [12.95, 14.72] | 7185 | 161 |
| test | Fine-tuned NMS-free | Ground plane | 19.96% [18.02, 22.06] | 17.66% | 19.14% [17.21, 21.18] | 7185 | 161 |
| test | Fine-tuned NMS-free | MLP ensemble | 8.93% [8.08, 9.97] | 6.01% | 8.66% [7.83, 9.71] | 7185 | 161 |

## temporal

| Partition | Detector | Speed rows | Speed MAE raw → filtered (m/s) | TTC within 20%, raw → filtered | TTC rows | Matched rows with history |
|---|---|---|---|---|---|---|
| development | COCO NMS | 23365 | 6.64 → 2.65 | 23.95% → 42.07% | 13975 | 94.75% |
| development | Fine-tuned NMS | 24138 | 6.60 → 2.86 | 22.23% → 35.99% | 14775 | 93.96% |
| development | Fine-tuned NMS-free | 20193 | 5.82 → 3.08 | 24.47% → 33.79% | 12390 | 89.58% |
| test | COCO NMS | 7278 | 5.99 → 1.99 | 25.80% → 45.37% | 4593 | 95.26% |
| test | Fine-tuned NMS | 7747 | 6.01 → 2.31 | 24.33% → 44.68% | 5125 | 94.76% |
| test | Fine-tuned NMS-free | 6802 | 5.38 → 2.59 | 25.60% → 38.09% | 4558 | 91.31% |

## classes

| Detector | Class | Object-frames | Tracks | Depth AbsRel [95% CI] | Range AbsRel [95% CI] |
|---|---|---|---|---|---|
| COCO NMS | Car | 5034 | 116 | 9.81% [8.50, 11.30] | 9.51% [8.15, 11.06] |
| COCO NMS | Van | 0 | 0 | — | — |
| COCO NMS | LargeVehicle | 55 | 1 | 22.74% [22.74, 22.74] | 21.93% [21.93, 21.93] |
| COCO NMS | Pedestrian | 2203 | 34 | 7.62% [6.49, 8.88] | 7.62% [6.52, 8.83] |
| COCO NMS | Cyclist | 0 | 0 | — | — |
| Fine-tuned NMS | Car | 5492 | 116 | 9.90% [8.72, 11.29] | 9.69% [8.52, 11.10] |
| Fine-tuned NMS | Van | 225 | 8 | 12.32% [9.05, 16.00] | 10.93% [7.85, 14.41] |
| Fine-tuned NMS | LargeVehicle | 0 | 0 | — | — |
| Fine-tuned NMS | Pedestrian | 1969 | 31 | 6.98% [5.74, 8.46] | 6.93% [5.73, 8.34] |
| Fine-tuned NMS | Cyclist | 183 | 6 | 5.04% [3.33, 9.35] | 5.07% [3.34, 9.36] |
| Fine-tuned NMS-free | Car | 5033 | 115 | 9.70% [8.61, 10.95] | 9.35% [8.28, 10.60] |
| Fine-tuned NMS-free | Van | 161 | 7 | 12.07% [7.85, 17.41] | 10.98% [7.60, 15.63] |
| Fine-tuned NMS-free | LargeVehicle | 1 | 1 | 52.43% [52.43, 52.43] | 44.40% [44.40, 44.40] |
| Fine-tuned NMS-free | Pedestrian | 1834 | 32 | 6.90% [5.52, 8.47] | 6.89% [5.54, 8.41] |
| Fine-tuned NMS-free | Cyclist | 156 | 6 | 4.78% [3.69, 7.79] | 4.75% [3.71, 7.65] |

## recall

| Detector | Class | 0–10 m | 10–20 m | 20–40 m | >40 m |
|---|---|---|---|---|---|
| COCO NMS | Car | 92.9% (902) | 91.2% (1391) | 69.4% (3208) | 52.8% (1361) |
| COCO NMS | Van | 0.0% (52) | 0.0% (61) | 0.0% (123) | 0.0% (201) |
| COCO NMS | LargeVehicle | 81.8% (11) | 100.0% (11) | 97.2% (36) | — |
| COCO NMS | Pedestrian | 75.2% (932) | 78.1% (1391) | 57.0% (726) | 7.1% (28) |
| COCO NMS | Cyclist | 0.0% (77) | 0.0% (107) | 0.0% (183) | 0.0% (6) |
| Fine-tuned NMS | Car | 93.3% (902) | 95.8% (1391) | 80.5% (3208) | 55.9% (1361) |
| Fine-tuned NMS | Van | 90.4% (52) | 96.7% (61) | 73.2% (123) | 15.9% (201) |
| Fine-tuned NMS | LargeVehicle | 0.0% (11) | 0.0% (11) | 0.0% (36) | — |
| Fine-tuned NMS | Pedestrian | 82.0% (932) | 73.0% (1391) | 26.0% (726) | 0.0% (28) |
| Fine-tuned NMS | Cyclist | 80.5% (77) | 74.8% (107) | 22.4% (183) | 0.0% (6) |
| Fine-tuned NMS-free | Car | 87.4% (902) | 91.5% (1391) | 73.3% (3208) | 47.3% (1361) |
| Fine-tuned NMS-free | Van | 80.8% (52) | 88.5% (61) | 44.7% (123) | 6.5% (201) |
| Fine-tuned NMS-free | LargeVehicle | 9.1% (11) | 0.0% (11) | 0.0% (36) | — |
| Fine-tuned NMS-free | Pedestrian | 73.4% (932) | 71.4% (1391) | 21.6% (726) | 0.0% (28) |
| Fine-tuned NMS-free | Cyclist | 66.2% (77) | 71.0% (107) | 15.8% (183) | 0.0% (6) |

## latency

| Pipeline | Decode (ms) | Detector forward (ms) | Distance/tracking/TTC (ms) | Serial total (ms) | Serial FPS | Decode-overlapped FPS |
|---|---|---|---|---|---|---|
| COCO NMS | 11.31 | 9.53 | 1.74 | 28.33 | 34.8 | 64.4 |
| Fine-tuned NMS | 11.14 | 9.16 | 1.67 | 27.33 | 36.3 | 67.5 |
| Fine-tuned NMS-free | 11.11 | 9.56 | 1.64 | 26.88 | 36.7 | 69.7 |

## paired

| Partition | Compared with COCO NMS | Common object-frames | Tracks | MLP AbsRel difference [95% CI], percentage points |
|---|---|---|---|---|
| development | Fine-tuned NMS | 20312 | 603 | +1.17 [+0.06, +2.98] |
| test | Fine-tuned NMS | 6538 | 146 | +0.23 [-0.21, +0.67] |
| development | Fine-tuned NMS-free | 18162 | 593 | +1.17 [-0.14, +3.23] |
| test | Fine-tuned NMS-free | 6071 | 147 | +0.08 [-0.45, +0.57] |
