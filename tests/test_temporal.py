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
