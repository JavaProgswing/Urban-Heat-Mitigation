"""Does training across cities raise honest R2 on a held-out city quadrant?

For each target city: baseline = train on its 3 quadrants, test on the 4th.
Augmented = same target quadrants + ALL pixels of the OTHER cities. Also tests
LST-anomaly target (LST - per-city median) to remove cross-city offset.
Leave-one-city-out (LOCO) = train on others, predict the whole target.
"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.features import drivers
from src.features.drivers import PHYSICS_SIGNS, DRIVER_COLS
from src.data.align import align_stack, derive_drivers, _default_name

CITIES = ["new_delhi", "bengaluru", "lucknow"]


def build(aoi):
    cfg = load_config().override(name=aoi)
    paths = {s: str(cfg.path("raw") / _default_name(s, cfg))
             for s in ("landsat", "sentinel", "era5", "ghsl", "ghsl_h", "terrain")
             if (cfg.path("raw") / _default_name(s, cfg)).exists()}
    df = drivers.stack_to_frame(derive_drivers(align_stack(paths, ref="landsat")))
    return df


def make(cols):
    cons = tuple(PHYSICS_SIGNS.get(c, 0) for c in cols)
    return xgb.XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=4,
                            min_child_weight=8, gamma=0.1, reg_lambda=2.0,
                            tree_method="hist", monotone_constraints=cons)


def quad_masks(df):
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    rm, cm = np.median(r), np.median(c)
    return [(r >= rm) & (c >= cm), (r >= rm) & (c < cm),
            (r < rm) & (c >= cm), (r < rm) & (c < cm)]


def main():
    cols = [c for c in DRIVER_COLS]
    dfs = {a: build(a) for a in CITIES}
    for a, d in dfs.items():
        use = [c for c in cols if c in d.columns]
        print(f"{a}: {len(d):,} px, {len(use)} feats, "
              f"LST {d['LST'].min():.0f}-{d['LST'].max():.0f} (med {d['LST'].median():.0f})")
    use = [c for c in cols if c in dfs[CITIES[0]].columns]
    rng = np.random.default_rng(0)
    cap = 120_000

    def Xy(d, anomaly=False):
        X = d[use].to_numpy("float32")
        y = d["LST"].to_numpy("float32")
        if anomaly:
            y = y - np.median(y)
        return X, y

    print("\ntarget        baseline   +pooled(abs)  +pooled(anom)   LOCO(anom)")
    for tgt in CITIES:
        others = [c for c in CITIES if c != tgt]
        base, pab, pan, loco = [], [], [], []
        Xt, yt = Xy(dfs[tgt]); _, ytan = Xy(dfs[tgt], anomaly=True)
        for te in quad_masks(dfs[tgt]):
            tr = ~te
            ti = np.flatnonzero(tr)
            ti = rng.choice(ti, cap, replace=False) if ti.size > cap else ti
            # baseline: target-only
            base.append(r2_score(yt[te], make(use).fit(Xt[ti], yt[ti]).predict(Xt[te])))
            # pooled absolute
            Xo = np.vstack([Xt[ti]] + [Xy(dfs[o])[0][:cap] for o in others])
            yo = np.concatenate([yt[ti]] + [Xy(dfs[o])[1][:cap] for o in others])
            pab.append(r2_score(yt[te], make(use).fit(Xo, yo).predict(Xt[te])))
            # pooled anomaly
            Xa = np.vstack([Xt[ti]] + [Xy(dfs[o], 1)[0][:cap] for o in others])
            ya = np.concatenate([ytan[ti]] + [Xy(dfs[o], 1)[1][:cap] for o in others])
            pan.append(r2_score(ytan[te], make(use).fit(Xa, ya).predict(Xt[te])))
        # LOCO anomaly: train others only, predict whole target (anomaly)
        Xo = np.vstack([Xy(dfs[o], 1)[0][:cap] for o in others])
        yo = np.concatenate([Xy(dfs[o], 1)[1][:cap] for o in others])
        loco = r2_score(ytan, make(use).fit(Xo, yo).predict(Xt))
        print(f"{tgt:12s}  {np.mean(base):.3f}      {np.mean(pab):.3f}        "
              f"{np.mean(pan):.3f}          {loco:.3f}")


if __name__ == "__main__":
    main()
