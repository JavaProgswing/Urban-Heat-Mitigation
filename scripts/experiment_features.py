"""Test which extra feature actually captures the smooth unexplained LST field
(resid autocorr ~0.96). Compares feature variants offline on cached tiles.

  base   : current 16 features (incl 210 m context)
  +large : add km-scale context means (450 m + 930 m) of land-cover drivers
  +coords: add normalised row/col (spatial position)  -- leakage check
  +both  : large context + coords

A good fix RAISES held-out R2 AND DROPS resid autocorr (captures real structure).
Coords may raise random-split R2 while barely denting autocorr on a spatial split
= memorising position, not physics.

    python scripts/experiment_features.py lucknow new_delhi
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.align import align_stack, derive_drivers, _nanmean_filter
from src.features import drivers
from src.features.drivers import split_xy, TARGET_COL
from src.models.train import _xgb_factory, _fit_early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import ndimage

LARGE = ("NDVI", "NDBI", "NDWI", "albedo", "build_frac")
SCALES = (15, 31)          # ~450 m, ~930 m at 30 m grid


def cached_paths(aoi):
    raw = ROOT / "data" / "raw"
    n = {"landsat": f"lst_landsat_{aoi}.tif", "sentinel": f"s2_lulc_{aoi}.tif",
         "era5": f"era5_{aoi}.tif", "ghsl": f"ghsl_{aoi}.tif",
         "terrain": f"terrain_{aoi}.tif"}
    return {k: str(raw / v) for k, v in n.items() if (raw / v).exists()}


def autocorr(r2d):
    m = np.isfinite(r2d)
    a = np.where(m, r2d, 0.0)
    num = ndimage.uniform_filter(a, size=3) * 9 - a
    den = ndimage.uniform_filter(m.astype("float32"), size=3) * 9 - m
    neigh = num / np.maximum(den, 1e-6)
    v = m & np.isfinite(neigh)
    return float(np.corrcoef(r2d[v], neigh[v])[0, 1])


def fit_variant(df, cols, shape, name):
    X = df[cols].to_numpy("float32")
    y = df[TARGET_COL].to_numpy("float32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    model = _fit_early_stopping(_xgb_factory(cols, True)(), Xtr, ytr, seed=0)
    r2 = r2_score(yte, model.predict(Xte))
    resid = y - model.predict(X)
    r2d = np.full(shape, np.nan, "float32")
    r2d[df["row"].to_numpy(), df["col"].to_numpy()] = resid
    print(f"  {name:9s} feat={len(cols):2d}  R2={r2:.4f}  autocorr={autocorr(r2d):.3f}")


def run(aoi):
    stack = derive_drivers(align_stack(cached_paths(aoi), ref="landsat"))
    shape = stack[TARGET_COL].shape
    # large-scale context grids
    for s in SCALES:
        for c in LARGE:
            if c in stack:
                stack[f"{c}_N{s}"] = _nanmean_filter(stack[c], s)
    # normalised coords
    H, W = shape
    yy, xx = np.indices(shape)
    stack["ROWN"] = (yy / H).astype("float32")
    stack["COLN"] = (xx / W).astype("float32")

    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0
    _, _, base = split_xy(df)
    large = [c for c in df.columns if "_N1" in c or "_N3" in c]
    coords = ["ROWN", "COLN"]

    print(f"\n=== {aoi} ===")
    fit_variant(df, base, shape, "base")
    fit_variant(df, base + large, shape, "+large")
    fit_variant(df, base + coords, shape, "+coords")
    fit_variant(df, base + large + coords, shape, "+both")


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["lucknow"]):
        run(a)
