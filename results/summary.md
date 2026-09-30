## selection

| val mAP50-95 | a: defaults | b: lower learning rate | c: + Van, Cyclist classes | d: c + DontCare grayed |
|---|---|---|---|---|
| fold 0 | 0.464 | **0.476** | 0.446 | 0.444 |
| fold 1 | 0.505 | **0.521** | 0.512 | 0.503 |
| fold 2 | **0.531** | 0.524 | 0.516 | 0.527 |

## detection

| Detector | Car AP50-95 | Pedestrian AP50-95 | mAP50-95 | mAP50 | mAP50-95 per fold |
|---|---|---|---|---|---|
| COCO weights | 0.503 ± 0.040 | 0.287 ± 0.105 | 0.395 ± 0.061 | 0.746 ± 0.056 | 0.462 / 0.379 / 0.344 |
| fine-tuned, NMS | 0.614 ± 0.033 | 0.315 ± 0.023 | 0.464 ± 0.024 | 0.790 ± 0.031 | 0.490 / 0.460 / 0.443 |
| fine-tuned, NMS-free | 0.574 ± 0.039 | 0.283 ± 0.023 | 0.429 ± 0.029 | 0.723 ± 0.027 | 0.455 / 0.434 / 0.397 |

## recall

| Detector | Class | 0-10 m | 10-20 m | 20-40 m | >40 m |
|---|---|---|---|---|---|
| COCO weights | Car | 90% (3264) | 87% (5464) | 77% (11853) | 61% (6718) |
| COCO weights | Pedestrian | 75% (3901) | 66% (5164) | 36% (2305) | 1% (100) |
| fine-tuned, NMS | Car | 93% (3264) | 90% (5464) | 82% (11853) | 64% (6718) |
| fine-tuned, NMS | Pedestrian | 80% (3901) | 60% (5164) | 23% (2305) | 0% (100) |
| fine-tuned, NMS-free | Car | 89% (3264) | 83% (5464) | 74% (11853) | 56% (6718) |
| fine-tuned, NMS-free | Pedestrian | 65% (3901) | 56% (5164) | 17% (2305) | 0% (100) |

## distance

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

## bins

| Detector | Method | Class | 0-10 m | 10-20 m | 20-40 m | >40 m |
|---|---|---|---|---|---|---|
| COCO weights | known size | Car | 46.8% (2901) | 9.8% (4729) | 8.4% (9088) | 7.4% (4067) |
| COCO weights | known size | Pedestrian | 11.1% (2937) | 7.1% (3384) | 8.0% (822) | 6.1% (1) |
| COCO weights | ground plane | Car | 31.9% (2901) | 14.9% (4729) | 24.5% (9088) | 39.3% (4067) |
| COCO weights | ground plane | Pedestrian | 13.7% (2937) | 16.7% (3384) | 32.6% (822) | 1.1% (1) |
| COCO weights | MLP | Car | 16.9% (2901) | 6.3% (4729) | 7.3% (9088) | 7.1% (4067) |
| COCO weights | MLP | Pedestrian | 8.6% (2937) | 6.2% (3384) | 6.5% (822) | 15.4% (1) |
| fine-tuned, NMS | known size | Car | 40.1% (2984) | 13.7% (4925) | 9.1% (9741) | 7.7% (4291) |
| fine-tuned, NMS | known size | Pedestrian | 9.4% (3135) | 5.5% (3112) | 9.4% (519) | - |
| fine-tuned, NMS | ground plane | Car | 39.4% (2984) | 16.9% (4925) | 25.5% (9741) | 36.1% (4291) |
| fine-tuned, NMS | ground plane | Pedestrian | 12.2% (3135) | 15.4% (3112) | 32.9% (519) | - |
| fine-tuned, NMS | MLP | Car | 27.6% (2984) | 6.4% (4925) | 6.7% (9741) | 6.8% (4291) |
| fine-tuned, NMS | MLP | Pedestrian | 8.6% (3135) | 5.8% (3112) | 6.7% (519) | - |
| fine-tuned, NMS-free | known size | Car | 39.5% (2868) | 13.9% (4509) | 9.5% (8712) | 7.6% (3782) |
| fine-tuned, NMS-free | known size | Pedestrian | 8.8% (2535) | 6.2% (2898) | 8.6% (400) | - |
| fine-tuned, NMS-free | ground plane | Car | 39.3% (2868) | 16.4% (4509) | 26.1% (8712) | 36.1% (3782) |
| fine-tuned, NMS-free | ground plane | Pedestrian | 11.0% (2535) | 15.1% (2898) | 36.3% (400) | - |
| fine-tuned, NMS-free | MLP | Car | 28.0% (2868) | 6.2% (4509) | 6.8% (8712) | 6.6% (3782) |
| fine-tuned, NMS-free | MLP | Pedestrian | 8.1% (2535) | 5.8% (2898) | 6.6% (400) | - |

## floor

| Objects found by | Method on the label boxes | All | Cars not truncated | Pedestrians not truncated |
|---|---|---|---|---|
| COCO weights | known size | 11.5% | 10.1% | 6.4% |
| COCO weights | ground plane | 25.2% | 26.4% | 15.3% |
| fine-tuned, NMS | known size | 12.0% | 10.1% | 6.9% |
| fine-tuned, NMS | ground plane | 25.5% | 26.2% | 14.9% |
| fine-tuned, NMS-free | known size | 12.1% | 10.2% | 6.6% |
| fine-tuned, NMS-free | ground plane | 25.3% | 25.9% | 14.7% |

## box_iou

| Detector | train (folds 0/1/2) | val | test |
|---|---|---|---|
| COCO weights | 0.821 / 0.813 / 0.804 | 0.776 / 0.839 / 0.857 | 0.837 / 0.804 / 0.805 |
| fine-tuned, NMS | 0.894 / 0.913 / 0.871 | 0.830 / 0.875 / 0.860 | 0.855 / 0.843 / 0.846 |
| fine-tuned, NMS-free | 0.895 / 0.913 / 0.874 | 0.832 / 0.879 / 0.850 | 0.862 / 0.853 / 0.841 |

## hypothesis

| Fold 0 | val mAP50-95, NMS head | val mAP50-95, NMS-free head | test mAP50-95, NMS head | test mAP50-95, NMS-free head | best epoch |
|---|---|---|---|---|---|
| a: defaults | 0.464 | 0.449 | 0.473 | 0.455 | 13 |
| e: epoch chosen by NMS-free head | 0.464 | 0.449 | 0.473 | 0.455 | 13 |

## latency

| Configuration | read | decode | preprocess | inference | postprocess | overhead | distance | total | total 95% CI | FPS per round |
|---|---|---|---|---|---|---|---|---|---|---|
| coco_1280 | 1.0 | 12.0 | 2.6 | 9.8 | 0.6 | 0.3 | 0.5 | 27.6 | 27.4-27.8 | 24/37/37 |
| coco_1280_nms | 0.9 | 11.9 | 3.4 | 9.5 | 1.4 | 0.3 | 0.5 | 27.9 | 27.4-28.0 | 36/36/36 |
| ft_1280 | 0.9 | 12.1 | 3.5 | 9.6 | 0.7 | 0.3 | 0.5 | 27.3 | 27.1-27.4 | 37/37/37 |
| ft_1280_cg | 0.9 | 11.1 | 2.0 | 9.3 | 0.5 | 0.3 | 0.5 | 24.6 | 24.5-24.7 | 43/41/39 |
| ft_1280_fp16 | 0.9 | 11.2 | 2.3 | 9.1 | 0.5 | 0.3 | 0.5 | 24.7 | 24.7-24.8 | 40/40/40 |
| ft_1280_nms | 0.9 | 11.8 | 3.3 | 9.2 | 1.4 | 0.3 | 0.5 | 27.4 | 27.1-27.6 | 37/37/37 |
| ft_1280_nms_cg | 0.9 | 11.1 | 2.3 | 9.4 | 1.3 | 0.3 | 0.5 | 26.0 | 25.9-26.0 | 38/38/38 |
| ft_1280_nms_pipelined | 0.8 | 11.2 | 2.4 | 8.6 | 1.3 | 0.3 | 0.5 | 13.2 | 13.2-13.3 | 74/74/74 |
| ft_1280_nms_trt16 | 0.9 | 11.0 | 2.3 | 3.8 | 1.2 | 0.3 | 0.5 | 20.1 | 20.1-20.1 | 49/49/50 |
| ft_1280_nms_trt16_pipelined | 0.8 | 11.0 | 2.4 | 3.9 | 1.3 | 0.3 | 0.5 | 12.2 | 12.2-12.3 | 80/81/79 |
| ft_1280_pipelined | 0.8 | 11.3 | 2.4 | 8.8 | 0.5 | 0.4 | 0.5 | 12.9 | 12.8-12.9 | 76/74/76 |
| ft_1280_trt16 | 0.9 | 11.1 | 2.3 | 3.9 | 0.5 | 0.3 | 0.5 | 19.4 | 19.4-19.5 | 51/51/51 |
| ft_1280_trt16_pipelined | 0.8 | 11.3 | 2.4 | 3.9 | 0.5 | 0.3 | 0.5 | 12.6 | 12.5-12.7 | 74/79/77 |
| ft_640 | 0.9 | 11.1 | 0.8 | 8.2 | 0.5 | 0.3 | 0.5 | 22.4 | 22.3-22.4 | 44/45/44 |
| ft_640_trt16 | 0.8 | 11.1 | 0.8 | 2.7 | 0.5 | 0.3 | 0.5 | 16.7 | 16.7-16.8 | 59/59/59 |
| ft_640_trt16_pipelined | 0.9 | 11.1 | 0.9 | 2.7 | 0.5 | 0.3 | 0.5 | 12.4 | 12.4-12.5 | 79/79/79 |
