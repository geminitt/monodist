"""Fine-tune all weights of the COCO-pretrained YOLO26n on the KITTI train sequences.

Ultralytics saves last.pt every epoch; rerunning the same command resumes an interrupted run.
The best epoch is chosen on the val sequences (Ultralytics fitness = 0.1 mAP50 + 0.9 mAP50-95).
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default="ram")
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
        exist_ok=True, seed=0, deterministic=True, plots=True,
    )


if __name__ == "__main__":
    main()
