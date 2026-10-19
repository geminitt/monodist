"""Image to obstacles: YOLO26n detections, the saved distance ensemble, depth Z and range of every object.

    python -m monodist.pipeline --weights best.pt --distance distance.pt --calib calib/0000.txt --images a.png b.png

The distance file (monodist.evaluate --save-model) carries the per-class score thresholds chosen with it, so the
detector and the distance model are used at the operating point they were evaluated at. The camera is given
by its focal length and principal point (f, cx, cy) in pixels; with a KITTI calibration file they come from P2.
"""
import argparse
import json

import numpy as np

from . import detect, geometry, mlp, temporal


def read_p2(path):
    """f, cx, cy of camera 2 from a KITTI calibration file."""
    for line in open(path):
        if line.startswith("P2:"):
            P2 = np.array(line.split()[1:], float).reshape(3, 4)
            return float(P2[0, 0]), float(P2[0, 2]), float(P2[1, 2])
    raise ValueError(f"no P2 in {path}")


class Pipeline:
    def __init__(self, weights, distance, imgsz=1280, nms=True, coco=False, half=False, temporal_settings=None):
        from ultralytics import YOLO
        self.detector = YOLO(str(weights))
        self.models, self.meta = mlp.load(distance, features=geometry.FEATURES)
        self.conf = np.array([self.meta["conf"][c] for c in self.meta["classes"]])
        self.imgsz, self.nms, self.coco, self.half = imgsz, nms, coco, half
        self.temporal_settings = temporal_settings or self.meta.get("temporal")
        self.reset()

    def reset(self):
        """Start a new sequence; tracking state must not carry across sequences or benchmark rounds."""
        self.tracker = temporal.Tracker(**self.temporal_settings) if self.temporal_settings else None
        self.frame = 0

    def distances(self, box, cls, camera, image_hw):
        """Depth Z and Euclidean range of each box, from the ensemble."""
        f, cx, cy = camera
        h, w = image_hw
        z = mlp.predict(self.models, geometry.features(box, cls, f, cx, cy, w, h)) if len(box) else np.zeros(0)
        return z, geometry.depth_to_range(box, z, f, cx, cy)

    def finish(self, r, camera):
        """Shared quality/timing path after detection: thresholds, ensemble, range and optional tracking."""
        keep = r["score"] >= self.conf[r["cls"]]
        d = {k: r[k][keep] for k in ("box", "score", "cls")}
        d["z"], d["range"] = self.distances(d["box"], d["cls"], camera, r["image_hw"])
        if self.tracker is not None:
            d["track"], d["filtered_z"], d["speed"] = self.tracker.update(self.frame, d["box"], d["cls"], d["z"])
            d["ttc"] = temporal.ttc(d["filtered_z"], d["speed"])
        self.frame += 1
        return d

    def __call__(self, images, camera):
        """One dict per image: box, score, cls (index into meta["classes"]), z and range of the kept objects."""
        out = []
        for r in detect.predict_images(self.detector, images, self.imgsz, float(self.conf.min()), self.coco,
                                       self.nms, self.half):
            out.append(self.finish(r, camera))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--distance", required=True, help="ensemble saved by monodist.evaluate --save-model")
    ap.add_argument("--images", nargs="+", required=True)
    cam = ap.add_mutually_exclusive_group(required=True)
    cam.add_argument("--calib", help="KITTI calibration file (P2)")
    cam.add_argument("--camera", nargs=3, type=float, metavar=("F", "CX", "CY"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--e2e", action="store_true", help="NMS-free head instead of the one-to-many head + NMS")
    ap.add_argument("--coco", action="store_true", help="COCO-trained weights")
    ap.add_argument("--temporal", help="JSON with meas_rel and accel chosen on validation; enables tracking/TTC")
    args = ap.parse_args()
    camera = read_p2(args.calib) if args.calib else tuple(args.camera)
    settings = json.load(open(args.temporal)) if args.temporal else None
    pipe = Pipeline(args.weights, args.distance, args.imgsz, nms=not args.e2e, coco=args.coco,
                    temporal_settings=settings)
    names = pipe.meta["classes"]
    for path, d in zip(args.images, pipe(args.images, camera)):
        objs = [{"class": names[c], "score": round(float(s), 3), "box": [round(float(v), 1) for v in b],
                 "z": round(float(z), 2), "range": round(float(r), 2)}
                for b, s, c, z, r in zip(d["box"], d["score"], d["cls"], d["z"], d["range"])]
        if "track" in d:
            for i, obj in enumerate(objs):
                obj.update(track=int(d["track"][i]), filtered_z=float(d["filtered_z"][i]),
                           speed=float(d["speed"][i]), ttc=float(d["ttc"][i]) if np.isfinite(d["ttc"][i]) else None)
        print(json.dumps({"image": path, "objects": objs}))


if __name__ == "__main__":
    main()
