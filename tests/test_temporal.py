import numpy as np

from monodist import temporal


def approaching(n=40, z0=40.0, v=-8.0, noise=0.07, seed=0):
    """An object closing at constant speed, measured with relative noise."""
    rng = np.random.default_rng(seed)
    frames = np.arange(n)
    z = z0 + v * frames * temporal.DT
    return frames, z, z * rng.normal(1, noise, n)


def test_filter_beats_raw_differences_on_speed():
    frames, z, zm = approaching()
    zf, vf = temporal.filter_track(frames, zm, meas_rel=0.07, accel=2.0)
    raw_v = np.diff(zm) / temporal.DT
    late = slice(20, None)  # after the filter has settled
    assert np.mean(np.abs(vf[late] + 8.0)) < 0.25 * np.mean(np.abs(raw_v[19:] + 8.0))
    assert np.mean(np.abs(zf[late] - z[late]) / z[late]) < np.mean(np.abs(zm[late] - z[late]) / z[late])


def test_filter_handles_frame_gaps():
    frames = np.array([0, 1, 2, 6, 7, 8])
    z = 30.0 - 5.0 * frames * temporal.DT
    zf, vf = temporal.filter_track(frames, z, meas_rel=0.01, accel=1.0)
    assert abs(vf[-1] + 5.0) < 0.5 and abs(zf[-1] - z[-1]) < 0.1


def test_ttc():
    assert np.allclose(temporal.ttc([20.0, 20.0, 20.0], [-10.0, 0.0, 3.0]), [2.0, np.inf, np.inf])


def test_tracker_keeps_identity_and_filters():
    tr = temporal.Tracker(meas_rel=0.05, accel=2.0)
    ids = []
    for f in range(10):
        box = np.array([[100 + f, 100, 150 + f, 150], [400, 100, 420, 160]], float)
        out_ids, zf, vf = tr.update(f, box, np.array([0, 1]), np.array([30.0 - 0.5 * f, 12.0]))
        ids.append(out_ids)
    ids = np.array(ids)
    assert (ids[:, 0] == ids[0, 0]).all() and (ids[:, 1] == ids[0, 1]).all() and ids[0, 0] != ids[0, 1]
    assert vf[0] < -2.0 and abs(vf[1]) < 1.0  # the first object approaches at 5 m/s, the second stands still


def test_tracker_does_not_join_different_classes_and_drops_old_tracks():
    tr = temporal.Tracker(meas_rel=0.05, accel=2.0, max_age=2)
    a, _, _ = tr.update(0, np.array([[0, 0, 10, 10.]]), np.array([0]), np.array([10.0]))
    b, _, _ = tr.update(1, np.array([[0, 0, 10, 10.]]), np.array([1]), np.array([10.0]))
    assert a[0] != b[0]
    c, _, _ = tr.update(10, np.array([[0, 0, 10, 10.]]), np.array([1]), np.array([10.0]))
    assert c[0] != b[0] and len(tr.tracks) == 1


def test_labeled_speed_of_a_real_track(root):
    speed = temporal.labeled_speed(root, ["0000"])
    assert len(speed) > 100 and all(np.isfinite(v) and abs(v) < 40 for v in speed.values())


def test_automatic_association_uses_no_ground_truth_and_resets_after_gaps():
    rows = {"seq": np.array(["0000"] * 4), "frame": np.array([0, 1, 2, 7]),
            "box": np.tile([0., 0., 10., 10.], (4, 1)), "cls": np.zeros(4, int),
            "pred_mlp": np.array([10., 100., 9., 8.]), "track": np.array([10, -1, 20, 10])}
    ids = temporal.associate(rows)
    assert ids[0] == ids[1] == ids[2] and ids[3] != ids[2]
    assert np.array_equal(ids, temporal.associate({**rows, "track": np.full(4, 42)}))
    # Replaying association once and filtering the predicted tracks must equal the live tracker.
    zf, vf = temporal.run_rows({**rows, "track": ids}, 0.02, 2.)
    live = temporal.Tracker(0.02, 2.)
    for i in range(4):
        ti, tz, tv = live.update(int(rows["frame"][i]), rows["box"][i:i + 1], rows["cls"][i:i + 1],
                                 rows["pred_mlp"][i:i + 1])
        assert ti[0] == ids[i] and np.allclose([tz[0], tv[0]], [zf[i], vf[i]])


def test_false_positives_enter_tracking_before_label_scoring():
    rows = {"seq": np.array(["0000"] * 3), "frame": np.arange(3),
            "box": np.tile([0., 0., 10., 10.], (3, 1)), "cls": np.zeros(3, int),
            "pred_mlp": np.array([10., 100., 9.]), "track": np.array([10, -1, 10]),
            "z": np.array([10., np.nan, 9.]), "status": np.array([1, 0, 1])}
    ids = temporal.associate(rows)
    speed = {("0000", 10, 0): -5., ("0000", 10, 2): -5.}
    result, _, _ = temporal.score_observations(rows, ids, speed, 0.01, 1.)
    assert result["matched_at_least_2m"] == 2 and result["n"] == 1
    assert np.isclose(result["raw"]["speed_mae"], 905.)  # the FP, not a label id, was the previous observation
    identity = temporal.association_summary(rows, ids)
    assert identity["unmatched_detections"] == 1
