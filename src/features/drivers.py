"""Assemble the driver feature stack (X) and LST target (y) as a tidy frame.

A 'driver' = a physical/morphological factor influencing LST. Columns:
  NDVI  vegetation cooling (evapotranspiration)
  NDBI  built-up / impervious heating
  NDWI  water presence (cooling)
  albedo  surface reflectivity (higher = cooler)
  build_frac  building footprint fraction (heat trapping)
  AIR_T, RH, WIND  atmospheric forcing (ERA5)
  POP  exposure weight (risk only, not a physical driver)
target:
  LST  land surface temperature (deg C)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ESA WorldCover class code -> name. Expanded to one-hot land-cover features
# (built/bare/tree/water...) in align.derive_drivers: categorical land cover is
# physically tied to LST and transfers across regions, adding signal the
# continuous indices blur (crop vs grass vs bare). Directly serves the PS
# objective "land use/land cover" as a quantified driver.
LULC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop",
                50: "built", 60: "bare", 80: "water"}
LULC_COLS = [f"LULC_{n}" for n in LULC_CLASSES.values()]

# Meteorology (ERA5 AIR_T/RH/WIND, ~11 km native) is deliberately EXCLUDED from
# the model. At city scale it spans only 1-5 pixels -> near-constant, so it adds
# no within-AOI signal, only a position proxy that overfits this scene and hurts
# transfer. Ablation on real tiles: dropping it lifted the honest spatial R2 from
# 0.42 -> 0.56 (Delhi) and 0.37 -> 0.43 (Bengaluru). ERA5 is still acquired (the
# atmospheric-driver pipeline) and kept in the frame for reporting; the model
# simply does not train on it.
# Neighbourhood context at two scales (align.spatial_drivers): `_N` ~210 m
# (immediate block) and `_NC` ~630 m (district). Landsat thermal is ~100 m
# native, so a pixel's LST reflects its surroundings, not just the 30 m cell.
# Dropping context CRASHES honest R2 (-> 0.19); the coarse scale adds ~+0.02.
_CONTEXT = ("NDVI", "NDBI", "NDWI", "albedo", "build_frac")
CONTEXT_COLS = [f"{c}_N" for c in _CONTEXT] + [f"{c}_NC" for c in _CONTEXT]

TEXTURE_COLS = ["NDVI_STD", "NDBI_STD", "albedo_STD", "BLD_H_STD"]

# Workflow-validated honest-R² lifts (adversarially re-checked under block CV on
# both Delhi and Lucknow): a ~1 km built-up context scale, and a dark-dense-built
# index. build_frac_NCC gave the largest single-feature gain (Delhi +0.031).
MORPH_COLS = ["build_frac_NCC", "DARKB"]

DRIVER_COLS = ["NDVI", "NDBI", "NDWI", "albedo", "build_frac", "BLD_H",
               "ELEV", "WATER_DIST"] + TEXTURE_COLS + CONTEXT_COLS + LULC_COLS \
              + MORPH_COLS
TARGET_COL = "LST"

# Physics priors on the sign of d(LST)/d(driver). Used as XGBoost monotonic
# constraints, so extrapolation to unseen
# intervention states (e.g. raised albedo) stays physically consistent.
#   -1 : more of this driver -> cooler ;  +1 : -> hotter ;  0 : unconstrained
PHYSICS_SIGNS = {
    "NDVI": -1,         # vegetation: evapotranspiration cooling
    "NDBI": +1,         # built-up / impervious: heating
    "NDWI": -1,         # water: cooling
    "albedo": -1,       # reflectivity: cooling
    "build_frac": +1,   # building density: heat trapping
    "BLD_H": 0,         # building height: daytime shading vs trapping -> ambiguous
    "ELEV": -1,         # higher elevation: cooler (lapse rate)
    "WATER_DIST": +1,   # farther from water: hotter (less cooling reach)
    "NDVI_STD": 0, "NDBI_STD": 0, "albedo_STD": 0, "BLD_H_STD": 0,  # texture
    "AIR_T": +1,        # warmer air -> warmer surface
    "RH": 0,            # ambiguous
    "WIND": -1,         # advective cooling
    # context features (both scales _N ~210 m and _NC ~630 m) inherit the sign
    # of their base driver so monotonic constraints hold at neighbourhood scale.
    "NDVI_N": -1, "NDVI_NC": -1,
    "NDBI_N": +1, "NDBI_NC": +1,
    "NDWI_N": -1, "NDWI_NC": -1,
    "albedo_N": -1, "albedo_NC": -1,
    "build_frac_N": +1, "build_frac_NC": +1,
    # morphology lifts (validated UNCONSTRAINED — a +1 leash fought the data and
    # erased the gain, since build_frac/albedo are already in the model; these
    # aren't intervention drivers and aren't refreshed in scenarios, so 0 is safe).
    "build_frac_NCC": 0,   # ~1 km built-up surroundings
    "DARKB": 0,            # dark dense built (build_frac*(1-albedo))
    # land-cover one-hots: presence of a hot class -> warmer (+1), cool class
    # -> cooler (-1); crop/shrub ambiguous (irrigation, density) -> unconstrained.
    "LULC_built": +1,
    "LULC_bare": +1,
    "LULC_tree": -1,
    "LULC_water": -1,
    "LULC_grass": -1,
    "LULC_crop": 0,
    "LULC_shrub": 0,
}


def stack_to_frame(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """Flatten aligned 2-D driver rasters + LST into a per-pixel DataFrame.

    arrays: {name: 2-D ndarray} all the same shape. Must include TARGET_COL.
    Adds pixel row/col so predictions can be re-gridded.
    """
    shape = arrays[TARGET_COL].shape
    rows, cols = np.indices(shape)
    data = {"row": rows.ravel(), "col": cols.ravel()}
    for name, arr in arrays.items():
        if arr.shape != shape:
            raise ValueError(f"{name} shape {arr.shape} != {shape}")
        data[name] = np.asarray(arr, dtype="float32").ravel()
    df = pd.DataFrame(data)
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def split_xy(df: pd.DataFrame):
    cols = [c for c in DRIVER_COLS if c in df.columns]
    return df[cols].to_numpy("float32"), df[TARGET_COL].to_numpy("float32"), cols


def hotspots(df: pd.DataFrame, pct: float = 90.0) -> pd.DataFrame:
    """Flag heat hotspots: pixels above the `pct` LST percentile."""
    thr = np.percentile(df[TARGET_COL], pct)
    out = df.copy()
    out["hotspot"] = (out[TARGET_COL] >= thr).astype(int)
    return out
