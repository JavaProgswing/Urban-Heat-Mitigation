"""Is the shipped 210 m context (_N) a real gain or random-split vanity?

Compares the original 11 raw drivers vs the current 16 (+ _N context) on BOTH
the optimistic random split and the leakage-proof spatial-block split, two cities.
Decides keep-vs-revert: a feature earns its place only if it holds (or at least
doesn't hurt) honest spatial R2.

    python scripts/final_verify.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.align import align_stack, derive_drivers
from src.features import drivers
from src.features.drivers import split_xy, TARGET_COL
from src.models.train import _xgb_factory, _fit_early_stopping, spatial_holdout_r2
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

ORIG11 = ["NDVI", "NDBI", "NDWI", "albedo", "build_frac", "ELEV",
          "WATER_DIST", "NDVI_STD", "AIR_T", "RH", "WIND"]


def cached_paths(aoi):
    raw = ROOT / "data" / "raw"
    n = {"landsat": f"lst_landsat_{aoi}.tif", "sentinel": f"s2_lulc_{aoi}.tif",
         "era5": f"era5_{aoi}.tif", "ghsl": f"ghsl_{aoi}.tif",
         "terrain": f"terrain_{aoi}.tif"}
    return {k: str(raw / v) for k, v in n.items() if (raw / v).exists()}


def rnd_r2(df, cols):
    X = df[cols].to_numpy("float32"); y = df[TARGET_COL].to_numpy("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    m = _fit_early_stopping(_xgb_factory(cols, True)(), Xtr, ytr, seed=0)
    return r2_score(yte, m.predict(Xte))


def run(aoi):
    stack = derive_drivers(align_stack(cached_paths(aoi), ref="landsat"))
    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0
    _, _, cur = split_xy(df)                      # current 16 (incl _N)
    print(f"\n=== {aoi} ===")
    for name, cols in [("orig-11", ORIG11), ("cur-16(+N)", cur)]:
        rr = rnd_r2(df, cols)
        sp = spatial_holdout_r2(_xgb_factory(cols, True), df, cols)
        print(f"  {name:11s} feat={len(cols):2d}  random={rr:.4f}  "
              f"honest(quadrant)={sp:.4f}")


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["lucknow", "new_delhi"]):
        run(a)
