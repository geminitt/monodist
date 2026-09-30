"""Distance error metrics and a cluster bootstrap that treats one tracked object as one sample."""
import numpy as np

BINS = np.array([0, 10, 20, 40, np.inf])
BIN_NAMES = ("0-10 m", "10-20 m", "20-40 m", ">40 m")


def abs_rel(pred, true):
    """|Zhat - Z| / Z, per object."""
    return np.abs(pred - true) / true


def cluster_bootstrap(values, clusters, stat=np.mean, n_boot=2000, seed=0):
    """95% percentile interval of stat(values), resampling whole clusters with replacement.

    values: (N,) or (N, K) array; clusters: (N,) ids. Frames of one track are strongly correlated,
    so resampling frames would pretend there are far more independent samples than there are.
    Returns (point estimate, low, high), each of shape values.shape[1:].
    """
    values = np.asarray(values, float)
    ids, inv = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    point = stat(values, axis=0)
    if stat is np.mean:  # fast path: a resample is a vector of cluster multiplicities
        counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n_boot)
        sums = np.zeros((len(ids),) + values.shape[1:])
        np.add.at(sums, inv, values)
        sizes = np.bincount(inv, minlength=len(ids)).astype(float)
        boots = np.tensordot(counts, sums, axes=1) / (counts @ sizes).reshape((-1,) + (1,) * (values.ndim - 1))
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        return point, lo, hi
    groups = [np.where(inv == k)[0] for k in range(len(ids))]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        boots.append(stat(values[idx], axis=0))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    return point, lo, hi


def summarize(pred, true, clusters, n_boot=2000):
    """Mean and median AbsRel with cluster-bootstrap intervals, plus the mean error in meters."""
    if len(true) == 0:
        return {"n": 0, "tracks": 0}
    e = abs_rel(pred, true)
    mean = cluster_bootstrap(e, clusters, np.mean, n_boot)
    med = cluster_bootstrap(e, clusters, np.median, n_boot)
    ids, inv = np.unique(clusters, return_inverse=True)
    per_track = np.bincount(inv, weights=e) / np.bincount(inv)
    track_mean = cluster_bootstrap(per_track, ids, np.mean, n_boot)
    return {
        "n": int(len(e)),
        "tracks": int(len(ids)),
        "absrel_mean": [float(v) for v in mean],
        "absrel_median": [float(v) for v in med],
        # every tracked object weighs the same, however many frames it appears in
        "absrel_track_mean": [float(v) for v in track_mean],
        "abs_m_mean": float(np.mean(np.abs(pred - true))),
    }


def paired_difference(err_a, err_b, clusters, n_boot=2000):
    """Mean of (err_a - err_b) on the same objects, with a cluster-bootstrap interval. Negative: a is better."""
    return [float(v) for v in cluster_bootstrap(err_a - err_b, clusters, np.mean, n_boot)]


def distance_bin(z):
    return np.searchsorted(BINS, z, side="right") - 1
