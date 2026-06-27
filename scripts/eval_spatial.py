"""Measure in-scene vs honest (spatial) R2 on cached real tiles — no GEE.

Builds the driver frame from data/raw/*.tif for an AOI, then reports the
random-split R2 and a 4-quadrant spatial-holdout R2 with the current model.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features import drivers
from src.data.align import align_stack, derive_drivers, _default_name
from src.models.train import _xgb_factory, _fit_early_stopping


def build_df(cfg):
    paths = {}
    for s in ("landsat", "sentinel", "era5", "ghsl", "ghsl_h", "terrain"):
        p = cfg.path("raw") / _default_name(s, cfg)
        if p.exists():
            paths[s] = str(p)
    stack = derive_drivers(align_stack(paths, ref="landsat"))
    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0
    return df


def quad_holdout(make, df, cols, max_n=80_000, seed=0):
    r, c = df["row"].to_numpy(), df["col"].to_numpy()
    rm, cm = np.median(r), np.median(c)
    quads = [(r >= rm) & (c >= cm), (r >= rm) & (c < cm),
             (r < rm) & (c >= cm), (r < rm) & (c < cm)]
    X = df[cols].to_numpy("float32"); y = df[drivers.TARGET_COL].to_numpy("float32")
    rng = np.random.default_rng(seed); out = []
    for te in quads:
        tr = ~te
        if te.sum() < 50 or tr.sum() < 50:
            continue
        ti = np.flatnonzero(tr)
        if ti.size > max_n:
            ti = rng.choice(ti, max_n, replace=False)
        m = make(); m.fit(X[ti], y[ti])
        out.append(r2_score(y[te], m.predict(X[te])))
    return out


def main():
    aoi = sys.argv[1] if len(sys.argv) > 1 else "new_delhi"
    cfg = load_config().override(name=aoi)
    df = build_df(cfg)
    X, y, cols = drivers.split_xy(df)
    print(f"AOI={aoi}  pixels={len(df):,}  features={len(cols)}")

    make = _xgb_factory(cols, physics=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = _fit_early_stopping(make(), Xtr, ytr)
    print(f"  in-scene R2   = {r2_score(yte, m.predict(Xte)):.3f}")

    q = quad_holdout(make, df, cols)
    print(f"  honest (single BR quadrant) = {q[0]:.3f}")
    print(f"  honest (4-quadrant mean)    = {np.mean(q):.3f}   "
          f"per-quad={[round(v, 2) for v in q]}")


if __name__ == "__main__":
    main()
