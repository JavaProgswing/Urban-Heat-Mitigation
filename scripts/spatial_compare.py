"""Honest spatial-block R2 per feature variant (the leakage-proof test).

Random-split R2 can't tell a transferable physical feature from position memorising.
This refits under spatial block hold-out so we keep only what generalises to
unseen ground. Expectation: +large (land-cover aggregates) holds; +coords (raw
row/col) collapses -- absolute position doesn't exist on a held-out tile.

    python scripts/spatial_compare.py lucknow
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.align import align_stack, derive_drivers, _nanmean_filter
from src.features import drivers
from src.features.drivers import split_xy
from src.models.train import _xgb_factory, spatial_block_cv

LARGE = ("NDVI", "NDBI", "NDWI", "albedo", "build_frac")
SCALES = (15, 31)


def cached_paths(aoi):
    raw = ROOT / "data" / "raw"
    n = {"landsat": f"lst_landsat_{aoi}.tif", "sentinel": f"s2_lulc_{aoi}.tif",
         "era5": f"era5_{aoi}.tif", "ghsl": f"ghsl_{aoi}.tif",
         "terrain": f"terrain_{aoi}.tif"}
    return {k: str(raw / v) for k, v in n.items() if (raw / v).exists()}


def run(aoi):
    stack = derive_drivers(align_stack(cached_paths(aoi), ref="landsat"))
    shape = stack["LST"].shape
    for s in SCALES:
        for c in LARGE:
            if c in stack:
                stack[f"{c}_N{s}"] = _nanmean_filter(stack[c], s)
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

    print(f"\n=== {aoi}  (honest spatial-block R2, n_blocks=3) ===")
    for name, cols in [("base", base), ("+large", base + large),
                       ("+coords", base + coords),
                       ("+both", base + large + coords)]:
        r2 = spatial_block_cv(_xgb_factory(cols, True), df, cols, n_blocks=3)
        print(f"  {name:9s} feat={len(cols):2d}  spatial R2={r2:.4f}")


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["lucknow"]):
        run(a)
