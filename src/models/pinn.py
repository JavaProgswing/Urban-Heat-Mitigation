"""Physics-Informed Neural Network (PINN) for land surface temperature.

Data loss : MSE(pred LST, observed LST).
Physics loss: soft penalties encoding surface-energy-balance priors, so the net
generalizes to unseen intervention states (cool roofs, greening) it never saw
in training:
  1. Albedo monotonicity   dLST/d(albedo) <= 0   (more reflective -> cooler)
  2. Vegetation cooling     dLST/d(NDVI)   <= 0   (ET cooling)
  3. Built-up heating       dLST/d(NDBI)   >= 0
Gradients come from autograd, so the constraints hold pointwise, not just on
average. Lambda weights set the physics/data trade-off.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..features.drivers import PHYSICS_SIGNS


@dataclass
class PINNConfig:
    hidden: tuple[int, ...] = (64, 64, 32)
    lr: float = 3e-3
    epochs: int = 600
    physics_weight: float = 0.2     # global weight on physics penalty
    warmup_frac: float = 0.3        # ramp physics weight over first 30% epochs
    batch: int = 4096
    seed: int = 0                   # reproducible init + shuffling
    device: str = "cpu"


class PINN:
    """Thin sklearn-style wrapper over a torch MLP with physics penalties."""

    def __init__(self, feature_names: list[str], cfg: PINNConfig | None = None):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.cfg = cfg or PINNConfig()
        self.feature_names = feature_names
        self.idx = {n: i for i, n in enumerate(feature_names)}
        self._mu = self._sd = None
        self._ymu = self._ysd = None      # target normalization

        torch.manual_seed(self.cfg.seed)  # reproducible weight init
        layers, d = [], len(feature_names)
        for h in self.cfg.hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers).to(self.cfg.device)

    # --- scaling ---------------------------------------------------------
    def _scale(self, X):
        return (X - self._mu) / self._sd

    # --- physics penalty -------------------------------------------------
    # Gradients are d(normalized LST)/d(scaled driver). Both scalings are
    # positive, so the sign equals real d(LST)/d(driver) — constraints valid.
    def _physics_loss(self, Xs, weight: float):
        t = self.torch
        Xs = Xs.clone().requires_grad_(True)
        y = self.net(Xs)
        grad = t.autograd.grad(y.sum(), Xs, create_graph=True)[0]
        pen = Xs.new_zeros(())
        for name, sign in PHYSICS_SIGNS.items():     # generic over all drivers
            if sign == 0 or name not in self.idx:
                continue
            g = grad[:, self.idx[name]]
            # want sign*g >= 0. Violation = -sign*g > 0. Penalty = relu(-sign*g).
            pen = pen + t.clamp(-sign * g, min=0.0).mean()
        return weight * pen

    # --- fit -------------------------------------------------------------
    def fit(self, X, y):
        t = self.torch
        X = np.asarray(X, "float32")
        y = np.asarray(y, "float32").reshape(-1, 1)
        self._mu = X.mean(0, keepdims=True)
        self._sd = X.std(0, keepdims=True) + 1e-6
        self._ymu = float(y.mean())
        self._ysd = float(y.std()) + 1e-6
        Xs = t.tensor(self._scale(X), device=self.cfg.device)
        yt = t.tensor((y - self._ymu) / self._ysd, device=self.cfg.device)

        t.manual_seed(self.cfg.seed)      # reproducible shuffling
        opt = t.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, self.cfg.epochs)
        mse = t.nn.MSELoss()
        n = Xs.shape[0]
        warm = max(1, int(self.cfg.warmup_frac * self.cfg.epochs))
        for ep in range(self.cfg.epochs):
            w = self.cfg.physics_weight * min(1.0, ep / warm)   # ramp physics
            perm = t.randperm(n)
            for i in range(0, n, self.cfg.batch):
                b = perm[i:i + self.cfg.batch]
                opt.zero_grad()
                pred = self.net(Xs[b])
                loss = mse(pred, yt[b]) + self._physics_loss(Xs[b], w)
                loss.backward()
                t.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            sched.step()
        return self

    # --- predict ---------------------------------------------------------
    def predict(self, X):
        t = self.torch
        X = np.asarray(X, "float32")
        Xs = t.tensor(self._scale(X), device=self.cfg.device)
        with t.no_grad():
            yn = self.net(Xs).cpu().numpy().ravel()
        return yn * self._ysd + self._ymu        # invert target normalization

    def save(self, path):
        self.torch.save(
            {"state": self.net.state_dict(), "mu": self._mu, "sd": self._sd,
             "ymu": self._ymu, "ysd": self._ysd,
             "features": self.feature_names}, str(path))

    def load(self, path):
        ck = self.torch.load(str(path), map_location=self.cfg.device)
        self.net.load_state_dict(ck["state"])
        self._mu, self._sd = ck["mu"], ck["sd"]
        self._ymu, self._ysd = ck["ymu"], ck["ysd"]
        self.feature_names = ck["features"]
        return self
