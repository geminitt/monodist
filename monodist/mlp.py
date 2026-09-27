"""The learned distance model: a small MLP from box features to log Z, trained from scratch on CPU."""
import numpy as np
import torch
from torch import nn


class DistanceMLP(nn.Module):
    def __init__(self, n_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)


def _fit(x, logz, epochs, seed, lr=1e-3, batch=512, x_eval=None, logz_eval=None):
    """Train for a fixed number of epochs; optionally record the L1 loss on an evaluation set after each epoch."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DistanceMLP(x.shape[1])
    model.mu.copy_(torch.tensor(x.mean(0), dtype=torch.float32))
    model.sd.copy_(torch.tensor(np.maximum(x.std(0), 1e-6), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    X, Y = torch.tensor(x, dtype=torch.float32), torch.tensor(logz, dtype=torch.float32)
    curve = []
    for _ in range(epochs):
        model.train()
        for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
            loss = (model(X[idx]) - Y[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if x_eval is not None:
            curve.append(float((predict_log(model, x_eval) - logz_eval).__abs__().mean()))
    return model, curve


@torch.no_grad()
def predict_log(model, x):
    model.eval()
    return model(torch.tensor(x, dtype=torch.float32)).numpy()


def predict(model, x):
    return np.exp(predict_log(model, x))


def train(x, z, groups, max_epochs=150, seed=0):
    """Pick the epoch count by leave-one-group-out (groups = sequences), then refit on everything.

    Returns the model, the chosen epoch count and the mean held-out L1 curve (in log Z).
    """
    torch.set_num_threads(2)  # a 5k-parameter model gains nothing from more threads
    logz = np.log(z)
    curves = []
    for g in np.unique(groups):
        held = groups == g
        _, curve = _fit(x[~held], logz[~held], max_epochs, seed, x_eval=x[held], logz_eval=logz[held])
        curves.append(curve)
    mean_curve = np.mean(curves, axis=0)
    best = int(np.argmin(mean_curve)) + 1
    model, _ = _fit(x, logz, best, seed)
    return model, best, mean_curve
