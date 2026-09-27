import numpy as np

from monodist import match


def frame(boxes, cls, neighbour=(), neighbour_cls=(), dontcare=()):
    n = len(boxes)
    return {"box": np.array(boxes, float).reshape(-1, 4), "cls": np.array(cls, int), "track": np.arange(n),
            "z": np.full(n, 10.0), "trunc": np.zeros(n, int), "occ": np.zeros(n, int),
            "neighbour_box": np.array(neighbour, float).reshape(-1, 4), "neighbour_cls": np.array(neighbour_cls, int),
            "dontcare": np.array(dontcare, float).reshape(-1, 4)}


def test_match_frame_statuses():
    g = frame([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 1], neighbour=[[40, 0, 50, 10]], neighbour_cls=[0],
              dontcare=[[60, 0, 80, 20]])
    det_box = np.array([[0, 0, 10, 10],      # hits car 0
                        [0, 0, 10, 10],      # duplicate of the same car -> FP
                        [20, 0, 30, 10],     # right place, wrong class (car on a pedestrian) -> FP
                        [40, 0, 50, 10],     # car on a Van -> ignored
                        [62, 2, 70, 10],     # inside DontCare -> ignored
                        [100, 0, 110, 10]])  # nothing there -> FP
    det_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    det_cls = np.array([0, 0, 0, 0, 0, 0])
    status, idx = match.match_frame(det_box, det_score, det_cls, g, thresholds=[0.5])
    assert status[:, 0].tolist() == [match.TP, match.FP, match.FP, match.IGNORED, match.IGNORED, match.FP]
    assert idx[0, 0] == 0


def test_greedy_matching_follows_score_order():
    g = frame([[0, 0, 10, 10]], [0])
    status, _ = match.match_frame(np.array([[1, 0, 11, 10], [0, 0, 10, 10]]), np.array([0.9, 0.3]),
                                  np.array([0, 0]), g, thresholds=[0.5])
    assert status[:, 0].tolist() == [match.TP, match.FP]  # the higher score wins even with the worse box


def test_iou_threshold_sweep():
    g = frame([[0, 0, 10, 10]], [0])
    status, _ = match.match_frame(np.array([[0, 0, 10, 8]]), np.array([0.9]), np.array([0]), g)
    iou_value = 0.8
    assert (status[0] == np.where(match.IOU_THRESHOLDS <= iou_value, match.TP, match.FP)).all()


def test_average_precision_known_cases():
    assert match.average_precision(np.array([0.9, 0.8]), np.array([1, 1]), 2) == 1.0
    assert np.isclose(match.average_precision(np.array([0.9]), np.array([1]), 2), 51 / 101)
    # FP ranked first: precision 1/2 at recall 1
    assert np.isclose(match.average_precision(np.array([0.9, 0.8]), np.array([0, 1]), 1), 0.5)
    # ignored detections do not count
    assert match.average_precision(np.array([0.9, 0.8]), np.array([-1, 1]), 1) == 1.0


def test_average_precision_close_to_ultralytics():
    from ultralytics.utils.metrics import compute_ap
    rng = np.random.default_rng(0)
    score = rng.random(300)
    tp = (rng.random(300) < score).astype(int)
    n_gt = int(tp.sum()) + 20
    order = np.argsort(-score)
    ctp = np.cumsum(tp[order])
    recall, precision = ctp / n_gt, ctp / np.arange(1, 301)
    ref, _, _ = compute_ap(recall, precision)
    assert abs(match.average_precision(score, tp, n_gt) - ref) < 0.01


def test_evaluate_and_pairs():
    gt = {("0000", 0): frame([[0, 0, 10, 10]], [0]), ("0000", 1): frame([], [])}
    det = {"seq": np.array(["0000", "0000"]), "frame": np.array([0, 1]), "box": np.array([[0, 0, 10, 10], [5, 5, 9, 9.]]),
           "score": np.array([0.9, 0.95]), "cls": np.array([0, 1])}
    status, idx = match.evaluate(det, gt)
    assert status[:, 0].tolist() == [match.TP, match.FP]
    pairs = match.matched_pairs(det, gt, status, idx, conf=0.25)
    assert len(pairs["z"]) == 1 and pairs["z"][0] == 10.0 and pairs["seq"][0] == "0000"
    rep = match.detection_report(det, gt, status)
    assert rep["Car"]["AP50"] == 1.0 and rep["Car"]["n_gt"] == 1
