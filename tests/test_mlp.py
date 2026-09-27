import numpy as np

from monodist import mlp


def test_mlp_learns_the_pinhole_relation():
    """Features h/f and class; target Z = H_class / (h/f) with 3% noise. The MLP should get close to the noise."""
    rng = np.random.default_rng(0)
    n = 4000
    cls = rng.integers(0, 2, n)
    z = rng.uniform(5, 60, n)
    H = np.where(cls == 0, 1.5, 1.75)
    x = np.stack([H / z * rng.normal(1, 0.03, n), cls.astype(float)], axis=1)
    groups = rng.integers(0, 4, n)
    model, epochs, curve = mlp.train(x[:3000], z[:3000], groups[:3000], max_epochs=60)
    pred = mlp.predict(model, x[3000:])
    assert np.median(np.abs(pred - z[3000:]) / z[3000:]) < 0.05
    assert 1 <= epochs <= 60 and len(curve) == 60
