"""Train + evaluate the LST model and quantify driver importance.

Baseline: gradient-boosted trees (fast, gives SHAP driver attribution).
Main: PINN (physics-constrained, used for scenario extrapolation).
Driver quantification answers objective #2 ("analyze drivers of urban heating").
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from ..features.drivers import split_xy, PHYSICS_SIGNS, TARGET_COL
from .pinn import PINN, PINNConfig


@dataclass
class TrainResult:
    model: object
    feature_names: list[str]
    metrics: dict          # mae, rmse, r2 on held-out test split
    importance: dict
    cv: dict | None = None  # k-fold mean/std of mae & r2 (robustness)
    eval: dict | None = None  # held-out y_true / y_pred for residual plots


def _metrics(y, pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def cross_validate(make_model, X, y, k=5, seed=0) -> dict:
    """k-fold CV -> mean/std of MAE and R2. make_model() returns a fresh model
    exposing .fit/.predict. Reports generalization, not single-split luck."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    maes, r2s = [], []
    for tr, te in kf.split(X):
        m = make_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        maes.append(mean_absolute_error(y[te], p))
        r2s.append(r2_score(y[te], p))
    return {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)), "k": k}


def spatial_block_cv(make_model, df, cols, n_blocks=4, seed=0) -> float | None:
    """Honest R2 under spatial block hold-out.

    A random pixel split lets near-identical neighbours land in both train and
    test, so the standard held-out R2 is optimistic (spatial leakage). Here we
    tile the grid into n_blocks x n_blocks contiguous blocks and hold out whole
    tiles, so test pixels are spatially separated from train. Reports the R2 the
    model would get on genuinely unseen ground."""
    r = df["row"].to_numpy()
    c = df["col"].to_numpy()
    edges = lambda v: np.quantile(v, np.linspace(0, 1, n_blocks + 1)[1:-1])
    block = np.digitize(r, edges(r)) * n_blocks + np.digitize(c, edges(c))
    X = df[cols].to_numpy("float32")
    y = df[TARGET_COL].to_numpy("float32")
    r2s = []
    for b in np.unique(block):
        te = block == b
        tr = ~te
        if te.sum() < 50 or tr.sum() < 50:
            continue
        m = make_model()
        m.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(r2s)) if r2s else None


def spatial_holdout_r2(make_model, df, cols, seed=0, max_n=80_000) -> float | None:
    """Honest spatial skill = mean R2 over a 2x2 quadrant cross-validation: hold
    out each quadrant in turn, train on the other three, average. Test pixels are
    spatially separated from train, so this does NOT reward the neighbour-leakage
    that inflates the random-split R2.

    Averaging all 4 quadrants (not just one corner) removes the high variance of a
    single arbitrary holdout — one atypical corner could otherwise dominate the
    reported number. Train side is subsampled to `max_n` rows per fold so the four
    extra fits stay a few seconds on 700k-pixel AOIs (diagnostic only)."""
    if "row" not in df or "col" not in df:
        return None
    r = df["row"].to_numpy()
    c = df["col"].to_numpy()
    rmed, cmed = np.median(r), np.median(c)
    quads = [(r >= rmed) & (c >= cmed), (r >= rmed) & (c < cmed),
             (r < rmed) & (c >= cmed), (r < rmed) & (c < cmed)]
    X = df[cols].to_numpy("float32")
    y = df[TARGET_COL].to_numpy("float32")
    rng = np.random.default_rng(seed)
    r2s = []
    for te in quads:
        tr = ~te
        if te.sum() < 50 or tr.sum() < 50:
            continue
        ti = np.flatnonzero(tr)
        if ti.size > max_n:
            ti = rng.choice(ti, max_n, replace=False)
        m = make_model()
        m.fit(X[ti], y[ti])
        r2s.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(r2s)) if r2s else None


def _subsample(mask: np.ndarray, n: int, rng) -> np.ndarray:
    """Return a boolean mask keeping a random n of the True entries of `mask`."""
    idx = np.flatnonzero(mask)
    keep = rng.choice(idx, n, replace=False)
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def _fit_early_stopping(model, Xtr, ytr, seed=0, rounds=50):
    """Fit with early stopping on an inner validation slice: grow many trees but
    keep only as many as keep improving held-out error -> better generalization
    than a fixed n_estimators. Falls back across xgboost API versions."""
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed)
    try:
        model.set_params(n_estimators=800, early_stopping_rounds=rounds)
        model.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
    except TypeError:
        try:                                   # older xgboost: fit-time kwarg
            model.set_params(n_estimators=800)
            model.fit(Xa, ya, eval_set=[(Xv, yv)],
                      early_stopping_rounds=rounds, verbose=False)
        except TypeError:                      # no early stopping available
            model.fit(Xtr, ytr)
    return model


def _xgb_factory(cols, physics):
    import xgboost as xgb
    constraints = (tuple(PHYSICS_SIGNS.get(c, 0) for c in cols)
                   if physics else None)
    # tree_method="hist": histogram-binned splits — ~5-10x faster than the exact
    # method on the 0.5-1M-pixel AOIs here, and supports monotone constraints.
    # n_jobs=4 (NOT -1): monotone constraints + all-cores hist thread-contend
    # badly in xgboost (≈14x slower on a many-core box); a small fixed pool is
    # dramatically faster. See scripts/eval_accuracy timing.
    # Regularized for spatial generalization, not just in-scene fit. LST is
    # highly autocorrelated, so an unregularized tree memorizes local texture
    # that doesn't transfer to unseen ground (inflates in-scene R2, not the
    # honest spatial R2). min_child_weight/gamma/reg_lambda prune that variance.
    return lambda: xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=4,
        min_child_weight=8, gamma=0.1, reg_lambda=2.0, reg_alpha=0.0,
        tree_method="hist", monotone_constraints=constraints,
    )


def train_xgb(df, test_size=0.2, seed=0, physics=True, cv=True,
              spatial_cv=False, max_train_rows=200_000) -> TrainResult:
    """Gradient-boosted LST model. physics=True applies monotonic constraints
    from PHYSICS_SIGNS so albedo/veg interventions cannot predict warming.
    Trees are early-stopped on an inner validation split. spatial_cv=True adds an
    honest block-hold-out R2 to metrics (slow: refits per block).

    On large AOIs the fit is capped at `max_train_rows`: adjacent LST pixels are
    near-identical (high spatial redundancy), so a random ~200k-pixel sample
    trains an equivalent model in a fraction of the time. Prediction, scenarios
    and maps still use every pixel — only the fit is subsampled."""
    X, y, cols = split_xy(df)
    if len(y) > max_train_rows:                 # subsample the redundant pixels
        idx = np.random.default_rng(seed).choice(len(y), max_train_rows,
                                                  replace=False)
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed)
    make = _xgb_factory(cols, physics)
    model = _fit_early_stopping(make(), Xtr, ytr, seed=seed)
    pred = model.predict(Xte)
    metrics = _metrics(yte, pred)
    # honest spatial skill (leakage-free). Cheap quadrant holdout for live use;
    # full block CV when explicitly asked (eval scripts).
    metrics["r2_spatial"] = spatial_holdout_r2(make, df, cols, seed=seed)
    if spatial_cv:
        metrics["r2_spatial"] = spatial_block_cv(make, df, cols, seed=seed)
    cv_res = cross_validate(make, X, y, seed=seed) if cv else None
    return TrainResult(model, cols, metrics,
                       _shap_importance(model, Xte, cols), cv=cv_res,
                       eval={"y_true": yte.tolist(), "y_pred": pred.tolist()})


def train_pinn(df, cfg: PINNConfig | None = None,
               test_size=0.2, seed=0) -> TrainResult:
    X, y, cols = split_xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed)
    model = PINN(cols, cfg).fit(Xtr, ytr)
    pred = model.predict(Xte)
    imp = _perm_importance(model, Xte, yte, cols)
    return TrainResult(model, cols, _metrics(yte, pred), imp,
                       eval={"y_true": yte.tolist(), "y_pred": pred.tolist()})


def _shap_importance(model, X, cols, max_rows=2000, seed=0):
    """Mean |SHAP| per feature. SHAP cost scales with rows, and importance is a
    row-mean, so a ~2000-row sample gives the same ranking far faster (TreeExplainer
    over a 40k-row test split is ~45s; over 2k it is a couple seconds)."""
    try:
        import shap
        if len(X) > max_rows:
            X = X[np.random.default_rng(seed).choice(len(X), max_rows, replace=False)]
        expl = shap.TreeExplainer(model)
        vals = np.abs(expl.shap_values(X)).mean(0)
        return dict(sorted(zip(cols, map(float, vals)),
                           key=lambda kv: -kv[1]))
    except Exception:
        return dict(zip(cols, map(float, model.feature_importances_)))


def _perm_importance(model, X, y, cols, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    base = mean_absolute_error(y, model.predict(X))
    out = {}
    for j, c in enumerate(cols):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(mean_absolute_error(y, model.predict(Xp)) - base)
        out[c] = float(np.mean(deltas))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
