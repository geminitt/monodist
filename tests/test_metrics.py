import numpy as np

from monodist import metrics


def test_abs_rel():
    assert np.allclose(metrics.abs_rel(np.array([9.0, 12.0]), np.array([10.0, 10.0])), [0.1, 0.2])


def test_constant_values_give_a_zero_width_interval():
    p, lo, hi = metrics.cluster_bootstrap(np.full(50, 0.3), np.arange(50) // 5)
    assert np.allclose([p, lo, hi], 0.3) and lo == hi


def test_duplicated_frames_do_not_narrow_the_interval():
    """100 frames of each of 20 tracks carry the information of 20 samples, not 2,000."""
    rng = np.random.default_rng(1)
    per_track = rng.normal(0, 1, 20)
    values = np.repeat(per_track, 100)
    clusters = np.repeat(np.arange(20), 100)
    _, lo_c, hi_c = metrics.cluster_bootstrap(values, clusters, n_boot=1000)
    _, lo_1, hi_1 = metrics.cluster_bootstrap(per_track, np.arange(20), n_boot=1000)
    _, lo_f, hi_f = metrics.cluster_bootstrap(values, np.arange(2000), n_boot=1000)
    assert abs((hi_c - lo_c) - (hi_1 - lo_1)) < 0.15 * (hi_1 - lo_1)
    assert (hi_f - lo_f) < 0.2 * (hi_c - lo_c)  # treating frames as independent would be ~10x too narrow


def test_distance_bins():
    assert metrics.distance_bin(np.array([0.5, 10.0, 19.9, 39.0, 80.0])).tolist() == [0, 1, 1, 2, 3]


def test_fast_mean_path_agrees_with_the_loop():
    rng = np.random.default_rng(3)
    values = rng.exponential(0.1, 600)
    clusters = rng.integers(0, 40, 600)
    fast = metrics.cluster_bootstrap(values, clusters, np.mean, n_boot=4000, seed=1)
    slow = metrics.cluster_bootstrap(values, clusters, lambda v, axis: np.mean(v, axis=axis), n_boot=4000, seed=1)
    assert np.allclose(fast, slow, atol=0.004)


def test_latency_bootstrap_median_and_imgsz_parsing():
    from monodist.detect import parse_imgsz
    from monodist.latency import bootstrap_median
    m, lo, hi = bootstrap_median(np.arange(101, dtype=float))
    assert m == 50 and lo < 50 < hi
    assert parse_imgsz("1280") == 1280 and parse_imgsz("416,1280") == [416, 1280]
