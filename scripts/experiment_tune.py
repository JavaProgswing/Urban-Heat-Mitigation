"""Honest-metric-guided XGBoost tuning. Picks hyperparameters by spatial-holdout
R2 (leakage-free), NOT random split, across multiple cities — so we only adopt
settings that genuinely generalise. More leaf regularisation (min_child_weight)
often helps on redundant/noisy LST pixels.

    python scripts/experiment_tune.py new_delhi lucknow
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.align import align_stack, derive_drivers
from src.features import drivers
from src.features.drivers import split_xy, PHYSICS_SIGNS, TARGET_COL
from src.models.train import spatial_holdout_r2
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

CONFIGS = {
    "base d6 mcw1":   dict(max_depth=6, min_child_weight=1,  learning_rate=0.05),
    "reg  d6 mcw20":  dict(max_depth=6, min_child_weight=20, learning_rate=0.05),
    "reg  d6 mcw50":  dict(max_depth=6, min_child_weight=50, learning_rate=0.05),
    "shallow d4 mcw10": dict(max_depth=4, min_child_weight=10, learning_rate=0.05),
    "deep d8 mcw20":  dict(max_depth=8, min_child_weight=20, learning_rate=0.05),
}


def cached_paths(aoi):
    raw = ROOT / "data" / "raw"
    n = {"landsat": f"lst_landsat_{aoi}.tif", "sentinel": f"s2_lulc_{aoi}.tif",
         "era5": f"era5_{aoi}.tif", "ghsl": f"ghsl_{aoi}.tif",
         "terrain": f"terrain_{aoi}.tif"}
    return {k: str(raw / v) for k, v in n.items() if (raw / v).exists()}


def make_factory(cols, cfg):
    cons = tuple(PHYSICS_SIGNS.get(c, 0) for c in cols)
    return lambda: xgb.XGBRegressor(
        n_estimators=400, subsample=0.8, colsample_bytree=0.8, n_jobs=4,
        tree_method="hist", monotone_constraints=cons, **cfg)


def run(aoi):
    stack = derive_drivers(align_stack(cached_paths(aoi), ref="landsat"))
    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0
    X, y, cols = split_xy(df)
    print(f"\n=== {aoi} (feat={len(cols)}) ===")
    for name, cfg in CONFIGS.items():
        make = make_factory(cols, cfg)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        m = make(); m.fit(Xtr, ytr)
        rr = r2_score(yte, m.predict(Xte))
        sp = spatial_holdout_r2(make, df, cols)
        print(f"  {name:18s} random={rr:.4f}  honest={sp:.4f}")


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["new_delhi", "lucknow"]):
        run(a)
