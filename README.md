# monodist — study notes

Personal learning material for [monodist](https://github.com/geminitt/monodist), kept on the orphan
branch `notes` so it stays out of the project's `main` history. The notebooks are written in
Vietnamese. Each one explains a concept from its definition, then checks it with code on the
project's own labels, detections and results. Everything runs on a CPU in under two minutes per
notebook.

Read them in order; each notebook only uses concepts introduced earlier.

| Notebook | Topic |
|---|---|
| `00_camera_lo_kim_va_kitti` | pinhole camera, intrinsics and the KITTI projection matrix, labels, depth vs distance, scale ambiguity |
| `01_hai_cong_thuc_khoang_cach` | known-size and ground-plane formulas, error propagation, the box-length effect, how flat KITTI roads are |
| `02_danh_gia_bo_nhan_dien` | IoU, greedy matching, KITTI ignore rules, precision and recall, AP, mAP50-95, F1 operating point |
| `03_yolo26_nms_va_fine_tune` | dense predictions and strides, NMS (implemented and checked), the one-to-one head, fine-tuning, overfitting signs, a rejected hypothesis |
| `04_mang_hoc_khoang_cach` | the distance MLP: log target, L1 vs L2, standardization, data leakage, leave-one-sequence-out, seed ensembles |
| `05_thong_ke_cho_ket_qua` | standard error, confidence intervals, bootstrap, intra-class correlation, cluster bootstrap, paired comparisons, the three-fold cross-validation |
| `06_do_do_tre` | latency vs throughput, GPU timing, Amdahl's law, kernel launches and CUDA graphs, fp16, TensorRT, PNG decoding, pipelining |

## Setup

This branch is checked out as a worktree inside the main checkout, so `..` is the project root:
the notebooks import `monodist` and read `../data/kitti_tracking` and `../results/`. The labels and
calibration are in the main repository; notebooks 00, 03 and 06 also need the 78 images of
sequence 0012, downloaded with the loop in `../data/kitti_tracking/README.md`.

```bash
git clone git@github.com:geminitt/monodist.git && cd monodist
git worktree add notes notes      # ./notes is ignored on main
pixi run jupyter lab notes/       # pixi finds ../pixi.toml
```

## Editing a notebook

Each notebook has a plain-text source in `src/`, with `### MD` and `### CODE` markers between
cells; edit the source, not the `.ipynb`. From this folder:

```bash
pixi run --manifest-path ../pixi.toml python tools/nbtool.py build src/00_camera_lo_kim_va_kitti.txt 00_camera_lo_kim_va_kitti.ipynb
pixi run --manifest-path ../pixi.toml python tools/nbtool.py dump 00_camera_lo_kim_va_kitti.ipynb
```

`build` writes the notebook and executes it, so stored outputs always match the code and results;
`sync` copies only the prose into an existing notebook and keeps its outputs.
