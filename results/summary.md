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
| development | COCO NMS | MLP ensemble | 8.42% [7.89, 9.02] | 5.83% | 8.11% [7.58, 8.68] | 22814 | 622 |
| development | Fine-tuned NMS | Known size | 14.27% [12.84, 16.36] | 10.38% | 13.21% [12.13, 14.64] | 24267 | 695 |
| development | Fine-tuned NMS | Ground plane | 25.37% [22.51, 28.33] | 17.18% | 24.33% [21.69, 27.00] | 24267 | 695 |
| development | Fine-tuned NMS | MLP ensemble | 9.43% [8.40, 10.60] | 5.48% | 8.83% [8.00, 9.74] | 24267 | 695 |
| development | Fine-tuned NMS-free | Known size | 14.39% [12.77, 16.73] | 10.62% | 13.25% [12.03, 14.81] | 21186 | 684 |
| development | Fine-tuned NMS-free | Ground plane | 24.99% [22.11, 28.38] | 16.80% | 23.86% [21.27, 26.73] | 21186 | 684 |
| development | Fine-tuned NMS-free | MLP ensemble | 9.10% [7.93, 10.62] | 5.21% | 8.50% [7.65, 9.47] | 21186 | 684 |
| test | COCO NMS | Known size | 13.30% [12.05, 14.71] | 9.12% | 12.31% [11.08, 13.62] | 7292 | 151 |
| test | COCO NMS | Ground plane | 21.94% [18.43, 25.96] | 17.34% | 21.25% [17.70, 25.29] | 7292 | 151 |
| test | COCO NMS | MLP ensemble | 9.56% [8.39, 11.01] | 6.59% | 9.37% [8.19, 10.84] | 7292 | 151 |
| test | Fine-tuned NMS | Known size | 14.89% [13.95, 15.90] | 12.41% | 14.03% [13.11, 14.93] | 7869 | 161 |
| test | Fine-tuned NMS | Ground plane | 20.68% [18.58, 23.02] | 17.76% | 19.84% [17.71, 22.13] | 7869 | 161 |
| test | Fine-tuned NMS | MLP ensemble | 9.17% [8.26, 10.28] | 6.17% | 8.91% [8.00, 10.06] | 7869 | 161 |
| test | Fine-tuned NMS-free | Known size | 14.64% [13.70, 15.62] | 12.00% | 13.82% [12.95, 14.72] | 7185 | 161 |
| test | Fine-tuned NMS-free | Ground plane | 19.96% [18.02, 22.06] | 17.66% | 19.14% [17.21, 21.18] | 7185 | 161 |
| test | Fine-tuned NMS-free | MLP ensemble | 9.08% [8.22, 10.09] | 6.18% | 8.78% [7.94, 9.81] | 7185 | 161 |

## temporal

| Partition | Detector | Speed rows | Speed MAE raw → filtered (m/s) | TTC within 20%, raw → filtered | TTC rows | Matched rows with history |
|---|---|---|---|---|---|---|
| development | COCO NMS | 23365 | 6.73 → 2.64 | 24.84% → 42.60% | 13975 | 94.75% |
| development | Fine-tuned NMS | 24138 | 6.83 → 2.84 | 23.05% → 36.54% | 14775 | 93.96% |
| development | Fine-tuned NMS-free | 20193 | 5.82 → 3.06 | 26.19% → 36.00% | 12390 | 89.58% |
| test | COCO NMS | 7278 | 5.96 → 1.96 | 25.63% → 46.44% | 4593 | 95.26% |
| test | Fine-tuned NMS | 7747 | 5.94 → 2.30 | 24.66% → 44.39% | 5125 | 94.76% |
| test | Fine-tuned NMS-free | 6802 | 5.51 → 2.60 | 25.87% → 37.82% | 4558 | 91.31% |

## classes

| Detector | Class | Object-frames | Tracks | Depth AbsRel [95% CI] | Range AbsRel [95% CI] |
|---|---|---|---|---|---|
| COCO NMS | Car | 5034 | 116 | 9.77% [8.34, 11.50] | 9.51% [7.98, 11.27] |
| COCO NMS | Van | 0 | 0 | — | — |
| COCO NMS | LargeVehicle | 55 | 1 | 19.64% [19.64, 19.64] | 18.45% [18.45, 18.45] |
| COCO NMS | Pedestrian | 2203 | 34 | 8.83% [7.19, 10.55] | 8.83% [7.22, 10.55] |
| COCO NMS | Cyclist | 0 | 0 | — | — |
| Fine-tuned NMS | Car | 5492 | 116 | 10.03% [8.88, 11.43] | 9.71% [8.53, 11.14] |
| Fine-tuned NMS | Van | 225 | 8 | 11.86% [9.06, 15.45] | 10.92% [7.98, 14.38] |
| Fine-tuned NMS | LargeVehicle | 0 | 0 | — | — |
| Fine-tuned NMS | Pedestrian | 1969 | 31 | 6.79% [5.59, 8.17] | 6.73% [5.61, 8.08] |
| Fine-tuned NMS | Cyclist | 183 | 6 | 5.75% [4.34, 9.64] | 5.82% [4.37, 9.80] |
| Fine-tuned NMS-free | Car | 5033 | 115 | 9.71% [8.68, 10.93] | 9.34% [8.31, 10.53] |
| Fine-tuned NMS-free | Van | 161 | 7 | 12.74% [8.72, 17.75] | 11.23% [8.05, 15.54] |
| Fine-tuned NMS-free | LargeVehicle | 1 | 1 | 89.51% [89.51, 89.51] | 79.54% [79.54, 79.54] |
| Fine-tuned NMS-free | Pedestrian | 1834 | 32 | 7.26% [5.82, 8.86] | 7.24% [5.84, 8.82] |
| Fine-tuned NMS-free | Cyclist | 156 | 6 | 5.68% [4.48, 8.66] | 5.72% [4.52, 8.74] |

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

| Pipeline | Decode (ms) | Detector forward (ms) | Distance/tracking/TTC (ms) | Serial total (ms) | Serial p99 [95% CI] (ms) | Serial FPS | Decode-overlapped FPS |
|---|---|---|---|---|---|---|---|
| COCO NMS | 12.30 | 10.22 | 2.02 | 31.13 | 37.61 [35.21, 42.58] | 31.8 | 51.6 |
| Fine-tuned NMS | 11.92 | 9.71 | 1.96 | 29.82 | 35.27 [33.86, 39.69] | 33.2 | 53.6 |
| Fine-tuned NMS-free | 11.99 | 10.27 | 1.95 | 29.60 | 34.70 [33.87, 37.44] | 33.6 | 54.6 |

## paired

| Partition | Compared with COCO NMS | Common object-frames | Tracks | MLP AbsRel difference [95% CI], percentage points |
|---|---|---|---|---|
| development | Fine-tuned NMS | 20312 | 603 | +0.27 [-0.45, +1.32] |
| test | Fine-tuned NMS | 6538 | 146 | -0.03 [-0.57, +0.46] |
| development | Fine-tuned NMS-free | 18162 | 593 | +0.37 [-0.62, +1.85] |
| test | Fine-tuned NMS-free | 6071 | 147 | -0.09 [-0.73, +0.43] |
