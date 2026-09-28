"""Fine-tune all weights of the COCO-pretrained YOLO26n on the train sequences of a split.

    python -m monodist.finetune --data kitti.yaml --project runs --name f0_a --config a

Ultralytics saves last.pt every epoch; rerunning the same command resumes an interrupted run. Within a run
the best epoch is chosen on the val sequences by Ultralytics' fitness (0.1 mAP50 + 0.9 mAP50-95).

Configurations compared on val in the cross-validation (see kaggle/cv):
  a  defaults: Car and Pedestrian labelled; optimizer "auto" picks AdamW, lr0 0.0017 (0.00125 with 4 classes)
  b  a with a lower learning rate (AdamW, lr0 0.0005)
  c  a with Van and Cyclist labelled as their own classes
  d  c with the DontCare regions painted grey
  e  a, but the epoch is chosen with the NMS-free (one-to-one) head instead of the NMS head
"""
import argparse
from pathlib import Path

CONFIGS = {
    "a": {},
    "b": {"train": {"optimizer": "AdamW", "lr0": 0.0005}},
    "c": {"extra_classes": True},
    "d": {"extra_classes": True, "mask_dontcare": True},
    "e": {"train": {"nms": False}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--config", default="a", choices=sorted(CONFIGS))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of training images (smoke tests)")
    args = ap.parse_args()

    from ultralytics import YOLO
    cache = {"false": False, "none": False, "true": True}.get(args.cache.lower(), args.cache)
    last = Path(args.project) / args.name / "weights" / "last.pt"
    if last.exists():
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as e:  # Ultralytics refuses to resume a finished run
            print("not resumed:", e)
        return
    YOLO(args.weights).train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
        workers=args.workers, cache=cache, fraction=args.fraction, project=args.project, name=args.name,
        exist_ok=True, seed=args.seed, deterministic=True, plots=True, **CONFIGS[args.config].get("train", {}),
    )


if __name__ == "__main__":
    main()
