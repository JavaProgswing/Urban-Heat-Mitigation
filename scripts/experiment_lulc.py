"""Does adding ESA WorldCover land-cover one-hots lift HONEST R2?

WorldCover (LULC) is already aligned in the stack but unused by the model. Land
cover is physically tied to LST and transferable (built=hot, tree/water=cool
everywhere), so unlike neighbourhood-context it should help the spatial-holdout
metric, not just random split. Verify before wiring into DRIVER_COLS.

    python scripts/experiment_lulc.py new_delhi lucknow
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

# WorldCover code -> name (major classes affecting LST)
LULC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop",
                50: "built", 60: "bare", 80: "water"}


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
    _, _, base = split_xy(df)

    lulc_cols = []
    if "LULC" in df:
        for code, name in LULC_CLASSES.items():
            col = f"LULC_{name}"
            df[col] = (df["LULC"].round().astype(int) == code).astype("float32")
            if df[col].mean() > 0.002:               # skip near-absent classes
                lulc_cols.append(col)

    print(f"\n=== {aoi} ===  LULC one-hots: {lulc_cols}")
    for name, cols in [("base", base), ("+LULC", base + lulc_cols)]:
        rr = rnd_r2(df, cols)
        sp = spatial_holdout_r2(_xgb_factory(cols, True), df, cols)
        print(f"  {name:7s} feat={len(cols):2d}  random={rr:.4f}  honest={sp:.4f}")


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["new_delhi", "lucknow"]):
        run(a)
