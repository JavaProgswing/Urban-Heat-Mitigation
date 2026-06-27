"""Diagnose whether low R2 is target noise (compositing artifacts) vs missing
drivers vs intrinsic difficulty. No GEE — works on cached tiles.

Reports, per AOI:
  - LST target stats + a stripe/seam score (thin high-gradient lines = mosaic
    seams from a multi-date median composite).
  - held-out R2 + residual spatial autocorrelation. Structured residuals (high
    autocorr) = a smooth signal the model misses or a smooth target artifact;
    white-noise residuals = pixel-level target noise (irreducible without
    cleaning y).
  - rural-ness: built_frac / NDBI level (low = peri-urban, lower LST ceiling).

    python scripts/verify_target.py lucknow new_delhi
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
from src.models.train import _xgb_factory, _fit_early_stopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import ndimage


def cached_paths(aoi: str) -> dict[str, str]:
    raw = ROOT / "data" / "raw"
    names = {"landsat": f"lst_landsat_{aoi}.tif", "sentinel": f"s2_lulc_{aoi}.tif",
             "era5": f"era5_{aoi}.tif", "ghsl": f"ghsl_{aoi}.tif",
             "terrain": f"terrain_{aoi}.tif"}
    return {k: str(raw / v) for k, v in names.items() if (raw / v).exists()}


def stripe_score(lst: np.ndarray) -> float:
    """Fraction of pixels on thin, long, high-gradient discontinuities (seams).
    Median composites of multi-date scenes leave straight seams; smooth natural
    LST gradients do not. Higher = more seam/stripe contamination."""
    a = np.where(np.isfinite(lst), lst, np.nan)
    gx = np.abs(np.gradient(a, axis=1))
    gy = np.abs(np.gradient(a, axis=0))
    g = np.fmax(gx, gy)
    thr = np.nanpercentile(g, 99)          # top-1% gradient = candidate edges
    edge = g >= thr
    # seams are spatially extended lines, not isolated specks: keep edge pixels
    # whose neighbourhood is also edge-rich
    dens = ndimage.uniform_filter(edge.astype("float32"), size=9)
    seam = edge & (dens > 0.15)
    return float(np.nanmean(seam))


def autocorr(resid2d: np.ndarray) -> float:
    """Lag-1 spatial autocorrelation of residuals (corr with neighbour mean)."""
    m = np.isfinite(resid2d)
    a = np.where(m, resid2d, 0.0)
    num = ndimage.uniform_filter(a, size=3) * 9 - a      # sum of 8 neighbours
    den = ndimage.uniform_filter(m.astype("float32"), size=3) * 9 - m
    neigh = num / np.maximum(den, 1e-6)
    v = m & np.isfinite(neigh)
    x, yv = resid2d[v], neigh[v]
    if x.size < 100:
        return float("nan")
    return float(np.corrcoef(x, yv)[0, 1])


def diagnose(aoi: str) -> None:
    paths = cached_paths(aoi)
    bands = align_stack(paths, ref="landsat")
    stack = derive_drivers(bands)
    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0
    X, y, cols = split_xy(df)

    lst2d = stack[TARGET_COL]
    ss = stripe_score(lst2d)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    model = _fit_early_stopping(_xgb_factory(cols, True)(), Xtr, ytr, seed=0)
    r2 = r2_score(yte, model.predict(Xte))

    # full-grid residuals -> 2-D for spatial structure
    pred_all = model.predict(X)
    resid = y - pred_all
    H, W = lst2d.shape
    r2d = np.full((H, W), np.nan, "float32")
    r2d[df["row"].to_numpy(), df["col"].to_numpy()] = resid
    ac = autocorr(r2d)
    resid_rmse = float(np.sqrt(np.mean(resid ** 2)))

    bf = df["build_frac"].mean() if "build_frac" in df else float("nan")
    ndbi = df["NDBI"].mean() if "NDBI" in df else float("nan")

    print(f"\n=== {aoi} ===")
    print(f"LST  range {np.nanmin(lst2d):.1f}..{np.nanmax(lst2d):.1f}  "
          f"std {np.nanstd(lst2d):.2f}  NaN {np.mean(~np.isfinite(lst2d)):.1%}")
    print(f"held-out R2        {r2:.4f}   resid RMSE {resid_rmse:.3f}")
    print(f"stripe/seam score  {ss:.4f}   (>0.01 = visible mosaic seams)")
    print(f"resid autocorr     {ac:.3f}   (hi=missing smooth driver/artifact, "
          f"~0=pixel target noise)")
    print(f"rural-ness         build_frac {bf:.3f}  NDBI {ndbi:+.3f}  "
          f"(low/neg = peri-urban)")


if __name__ == "__main__":
    for aoi in (sys.argv[1:] or ["lucknow"]):
        diagnose(aoi)
