"""Feature ablation vs honest 4-quadrant R2 on cached real tiles. Tests whether
dropping near-constant / position-proxy features improves generalization."""
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
from src.features.drivers import PHYSICS_SIGNS, DRIVER_COLS
from src.data.align import align_stack, derive_drivers, _default_name

MET = ["AIR_T", "RH", "WIND"]
CONTEXT = [c for c in DRIVER_COLS if c.endswith("_N")]
SETS = {
    "all (24)": DRIVER_COLS,
    "drop meteorology": [c for c in DRIVER_COLS if c not in MET],
    "drop ELEV+WATER_DIST": [c for c in DRIVER_COLS if c not in ("ELEV", "WATER_DIST")],
    "drop _N context": [c for c in DRIVER_COLS if c not in CONTEXT],
    "drop met+elev+wdist": [c for c in DRIVER_COLS if c not in MET + ["ELEV", "WATER_DIST"]],
}


def build_df(aoi):
    cfg = load_config().override(name=aoi)
    paths = {s: str(cfg.path("raw") / _default_name(s, cfg))
             for s in ("landsat", "sentinel", "era5", "ghsl", "ghsl_h", "terrain")
             if (cfg.path("raw") / _default_name(s, cfg)).exists()}
    return drivers.stack_to_frame(derive_drivers(align_stack(paths, ref="landsat")))


def make(cols):
    cons = tuple(PHYSICS_SIGNS.get(c, 0) for c in cols)
    return xgb.XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                            min_child_weight=8, gamma=0.1, reg_lambda=2.0,
                            tree_method="hist", monotone_constraints=cons)


def honest(df, cols):
    use = [c for c in cols if c in df.columns]
    X = df[use].to_numpy("float32"); y = df[drivers.TARGET_COL].to_numpy("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    ins = r2_score(yte, make(use).fit(Xtr, ytr).predict(Xte))
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    rm, cm = np.median(r), np.median(c)
    quads = [(r >= rm) & (c >= cm), (r >= rm) & (c < cm),
             (r < rm) & (c >= cm), (r < rm) & (c < cm)]
    rng = np.random.default_rng(0); hs = []
    for te in quads:
        ti = np.flatnonzero(~te)
        if ti.size > 80_000:
            ti = rng.choice(ti, 80_000, replace=False)
        hs.append(r2_score(y[te], make(use).fit(X[ti], y[ti]).predict(X[te])))
    return ins, float(np.mean(hs))


def main():
    cities = sys.argv[1:] or ["new_delhi", "bengaluru"]
    dfs = {a: build_df(a) for a in cities}
    print(f"{'feature set':22s} " + "  ".join(f"{a:>20s}" for a in cities))
    for name, cols in SETS.items():
        cells = [f"{honest(dfs[a], cols)[0]:.3f}/{honest(dfs[a], cols)[1]:.3f}"
                 for a in cities]
        print(f"{name:22s} " + "      ".join(f"{x:>14s}" for x in cells))


if __name__ == "__main__":
    main()
