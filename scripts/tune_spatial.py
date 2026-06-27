"""Grid a few XGB configs against the honest 4-quadrant spatial R2 on cached
real tiles. Picks settings that generalize to unseen ground, not just in-scene."""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.features import drivers
from src.data.align import align_stack, derive_drivers, _default_name
from src.features.drivers import PHYSICS_SIGNS

CONFIGS = {
    "current d6/mcw8/l2":   dict(max_depth=6, min_child_weight=8, gamma=0.1, reg_lambda=2.0, colsample_bytree=0.8),
    "d5/mcw16/l4":          dict(max_depth=5, min_child_weight=16, gamma=0.2, reg_lambda=4.0, colsample_bytree=0.7),
    "d4/mcw32/l6":          dict(max_depth=4, min_child_weight=32, gamma=0.3, reg_lambda=6.0, colsample_bytree=0.7),
    "d5/mcw32/l8/cs.6":     dict(max_depth=5, min_child_weight=32, gamma=0.3, reg_lambda=8.0, colsample_bytree=0.6),
    "d4/mcw64/l8/cs.6":     dict(max_depth=4, min_child_weight=64, gamma=0.4, reg_lambda=8.0, colsample_bytree=0.6),
}


def build_df(aoi):
    cfg = load_config().override(name=aoi)
    paths = {s: str(cfg.path("raw") / _default_name(s, cfg))
             for s in ("landsat", "sentinel", "era5", "ghsl", "ghsl_h", "terrain")
             if (cfg.path("raw") / _default_name(s, cfg)).exists()}
    df = drivers.stack_to_frame(derive_drivers(align_stack(paths, ref="landsat")))
    return df


def make(cols, **kw):
    cons = tuple(PHYSICS_SIGNS.get(c, 0) for c in cols)
    return xgb.XGBRegressor(n_estimators=350, learning_rate=0.05, subsample=0.8,
                            n_jobs=4, tree_method="hist", monotone_constraints=cons,
                            **kw)


def evaluate(df, cols, kw):
    X = df[cols].to_numpy("float32"); y = df[drivers.TARGET_COL].to_numpy("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    ins = r2_score(yte, make(cols, **kw).fit(Xtr, ytr).predict(Xte))
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    rm, cm = np.median(r), np.median(c)
    quads = [(r >= rm) & (c >= cm), (r >= rm) & (c < cm),
             (r < rm) & (c >= cm), (r < rm) & (c < cm)]
    rng = np.random.default_rng(0); hs = []
    for te in quads:
        tr = ~te; ti = np.flatnonzero(tr)
        if ti.size > 80_000:
            ti = rng.choice(ti, 80_000, replace=False)
        hs.append(r2_score(y[te], make(cols, **kw).fit(X[ti], y[ti]).predict(X[te])))
    return ins, float(np.mean(hs))


def main():
    cities = sys.argv[1:] or ["new_delhi", "bengaluru"]
    dfs = {a: build_df(a) for a in cities}
    cols = drivers.split_xy(dfs[cities[0]])[2]
    print(f"{'config':22s} " + "  ".join(f"{a:>22s}" for a in cities))
    print(f"{'':22s} " + "  ".join("in-scene / honest4q   " for _ in cities))
    for name, kw in CONFIGS.items():
        cells = []
        for a in cities:
            ins, hon = evaluate(dfs[a], cols, kw)
            cells.append(f"{ins:.3f} / {hon:.3f}        ")
        print(f"{name:22s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
