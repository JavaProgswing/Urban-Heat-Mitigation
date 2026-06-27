"""Align all exported GeoTIFFs onto one common grid -> driver stack.

This is the bridge between raw satellite/met exports and the model. Each source
is exported at its own native resolution/CRS; here we reproject-match every layer
to the Landsat LST grid (the modeling reference), extract the named bands, derive
albedo + build_frac, and return a dict of aligned 2-D arrays ready for
`features.drivers.stack_to_frame`.

The `align_stack` core is pure rioxarray (no Earth Engine) -> unit-testable.
`build_driver_stack` orchestrates export (needs GEE) + align + derive.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..config import Config
from ..features import indices
from ..features.drivers import TARGET_COL

# Band layout of each exported GeoTIFF = export order in the data modules.
BAND_LAYOUT = {
    "landsat": ["LST"],
    "sentinel": ["B2", "B3", "B4", "B8", "B11", "B12",
                 "NDVI", "NDBI", "NDWI", "ALBEDO", "LULC"],
    "ghsl": ["BUILT", "POP"],
    "ghsl_h": ["BUILT_H"],
    "era5": ["AIR_T", "RH", "WIND"],
    "terrain": ["ELEV"],
    "ecostress": ["LST_ECO"],          # cross-sensor validation only, not a driver
}


def _open(path):
    import rioxarray as rxr
    return rxr.open_rasterio(path, masked=True)


def align_stack(paths: dict[str, str | Path], ref: str = "landsat") -> dict:
    """Reproject every source raster to the reference grid; return named bands.

    paths: {source_name: geotiff_path} where source_name is a BAND_LAYOUT key.
    Returns {band_name: 2-D float32 ndarray}, all sharing the reference shape.
    """
    ref_da = _open(paths[ref])
    out: dict[str, np.ndarray] = {}
    try:
        for src, path in paths.items():
            da = _open(path)
            try:
                matched = da.rio.reproject_match(ref_da) if src != ref else da
                arr = np.asarray(matched.values, dtype="float32")  # (bands,H,W)
                if arr.ndim == 2:
                    arr = arr[None]
                for i, name in enumerate(BAND_LAYOUT[src]):
                    if i < arr.shape[0]:
                        out[name] = arr[i].copy()      # detach from file buffer
            finally:
                da.close()                             # release GDAL handle
    finally:
        ref_da.close()
    return out


def derive_drivers(bands: dict[str, np.ndarray]) -> dict:
    """Map raw bands to the model's driver schema: real albedo, build_frac,
    water-distance, NDVI texture (+ LST, POP, ELEV pass through)."""
    out = dict(bands)

    # albedo: prefer the real Sentinel-2 broadband band; else NDVI proxy.
    if "ALBEDO" in out:
        out["albedo"] = np.clip(out["ALBEDO"], 0.02, 0.7).astype("float32")
    elif "NDVI" in out and "albedo" not in out:
        out["albedo"] = indices.albedo_from_ndvi(out["NDVI"]).astype("float32")

    # build_frac: normalize GHSL built surface 0..1 by p99; else NDBI fallback.
    # Guarded so derive_drivers is idempotent (won't clobber an existing value).
    if "build_frac" not in out:
        if "BUILT" in out:
            b = out["BUILT"]
            p99 = np.nanpercentile(b, 99) or 1.0
            out["build_frac"] = np.clip(b / p99, 0, 1).astype("float32")
        elif "NDBI" in out:
            out["build_frac"] = np.clip((out["NDBI"] + 1) / 2, 0, 1).astype("float32")

    # building height (vertical morphology); 0 where no buildings / band absent
    if "BUILT_H" in out and "BLD_H" not in out:
        out["BLD_H"] = np.clip(np.nan_to_num(out["BUILT_H"], nan=0.0),
                               0, 200).astype("float32")

    # spatial drivers computed locally from the aligned grids (no extra GEE)
    out.update(spatial_drivers(out))
    out.update(lulc_onehot(out))
    return out


def lulc_onehot(out: dict[str, np.ndarray]) -> dict:
    """Expand the ESA WorldCover class band into per-class one-hot rasters
    (LULC_built, LULC_tree, ...). NaN class -> all-zero (no class)."""
    from ..features.drivers import LULC_CLASSES
    if "LULC" not in out:
        return {}
    codes = np.round(np.nan_to_num(out["LULC"], nan=-1)).astype("int32")
    return {f"LULC_{name}": (codes == code).astype("float32")
            for code, name in LULC_CLASSES.items()}


# Land-cover drivers aggregated to coarser windows as neighbourhood "context".
# LST is multi-scale: a pixel's temperature reflects its surroundings at several
# ranges. Two scales — 7 px (~210 m, immediate block; `_N`) and 21 px (~630 m,
# district; `_NC`). Ablation: adding the 21 px scale lifted honest spatial R2
# (Delhi 0.564 -> 0.583). A third, larger scale over-fit and was dropped.
CONTEXT_DRIVERS = ("NDVI", "NDBI", "NDWI", "albedo", "build_frac")
CONTEXT_SIZE = 7
CONTEXT_SIZE_COARSE = 21
# Local-texture (std) drivers: morphological heterogeneity raises honest R2.
TEXTURE_DRIVERS = ("NDVI", "NDBI", "albedo", "BLD_H")


def _nanmean_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Neighbourhood mean that ignores NaN, so masked pixels don't poison a whole
    window (a plain uniform_filter would NaN-out every cell touching a gap)."""
    from scipy import ndimage
    a = np.asarray(arr, dtype="float32")
    valid = np.isfinite(a)
    num = ndimage.uniform_filter(np.where(valid, a, 0.0), size=size, mode="nearest")
    den = ndimage.uniform_filter(valid.astype("float32"), size=size, mode="nearest")
    out = num / np.maximum(den, 1e-6)
    out[den < 1e-6] = np.nan
    return out.astype("float32")


def spatial_drivers(out: dict[str, np.ndarray]) -> dict:
    """Distance-to-water (from NDWI) + NDVI texture + neighbourhood-context means.
    Operate on the 2-D grid before flattening, so neighbourhood structure is
    preserved."""
    from scipy import ndimage
    extra: dict[str, np.ndarray] = {}

    if "NDWI" in out and "WATER_DIST" not in out:
        water = np.nan_to_num(out["NDWI"], nan=-1.0) > 0.0   # surface-water mask
        if water.any():
            dist = ndimage.distance_transform_edt(~water).astype("float32")
        else:                                           # no water in scene
            dist = np.full(out["NDWI"].shape, float(max(out["NDWI"].shape)),
                           dtype="float32")
        extra["WATER_DIST"] = dist

    # Local texture (5px std) of vegetation / built-up / albedo / height — urban
    # morphological heterogeneity is an LST driver (mixed surfaces vs uniform).
    # NaN-aware (plain uniform_filter propagates NaN -> on partial coverage it
    # would make a feature all-NaN and dropna() would wipe the whole frame).
    for c in TEXTURE_DRIVERS:
        if c in out and f"{c}_STD" not in out:
            a = out[c].astype("float32")
            mean = _nanmean_filter(a, 5)
            sq = _nanmean_filter(a * a, 5)
            extra[f"{c}_STD"] = np.sqrt(np.maximum(sq - mean * mean, 0)).astype("float32")

    # coarse-scale context at two ranges: lets the model see a pixel's
    # surroundings, matching LST's true multi-scale footprint vs the 30 m grid.
    for c in CONTEXT_DRIVERS:
        if c in out and f"{c}_N" not in out:
            extra[f"{c}_N"] = _nanmean_filter(out[c], CONTEXT_SIZE)
        if c in out and f"{c}_NC" not in out:
            extra[f"{c}_NC"] = _nanmean_filter(out[c], CONTEXT_SIZE_COARSE)

    return extra


def build_driver_stack(cfg: Config, reuse: bool = True) -> dict:
    """Export every dataset via GEE, align to Landsat grid, derive drivers.

    Resilient: a single failed source (e.g. GHSL) is skipped, not fatal — the
    pipeline proceeds with the remaining drivers (build_frac falls back to NDBI,
    POP to a constant). Landsat LST is the only hard requirement (it is the
    target + reference grid).

    reuse=True keeps an already-downloaded tile from a prior run if re-export
    fails, so a partial GEE outage doesn't lose good data.

    Returns {driver/target: 2-D ndarray} including TARGET_COL ('LST').
    """
    from . import landsat, sentinel, era5, ghsl, terrain
    exporters = {
        "landsat": landsat.export_lst,
        "sentinel": sentinel.export,
        "era5": era5.export,
        "ghsl": ghsl.export,
        "ghsl_h": ghsl.export_height,      # optional vertical morphology
        "terrain": terrain.export,
    }
    paths: dict[str, str] = {}
    for name, fn in exporters.items():
        cached = cfg.path("raw") / _default_name(name, cfg)
        got = _export_with_retry(name, fn, cfg, attempts=3)
        if got:
            paths[name] = got
        elif reuse and _valid(cached):
            print(f"  [{name}: reusing cached tile from a prior run]")
            paths[name] = str(cached)
        else:
            print(f"  [skip {name}: export failed, no cached tile]")

    if "landsat" not in paths:
        raise RuntimeError(
            "Landsat LST export failed — it is the target + reference grid, "
            "cannot proceed. Check GEE auth / project / AOI.")

    bands = align_stack(paths, ref="landsat")
    stack = derive_drivers(bands)
    if TARGET_COL not in stack:
        raise RuntimeError("LST band missing after alignment")
    missing = [d for d in ("AIR_T", "RH", "WIND", "build_frac")
               if d not in stack]
    if missing:
        print(f"  [drivers missing, model will train without them: {missing}]")
    return stack


def _valid(path) -> bool:
    """A usable GeoTIFF = exists and non-empty. geemap can fail to write a file
    yet not raise, so existence must be checked explicitly."""
    from pathlib import Path
    p = Path(path)
    return p.exists() and p.stat().st_size > 1024


def _export_with_retry(name, fn, cfg, attempts: int = 3):
    """Export, verifying a real file lands. Retries transient GEE/network
    failures (IncompleteRead, silent download errors). Returns path or None."""
    for i in range(attempts):
        try:
            out = fn(cfg)
            if _valid(out):
                return str(out)
            print(f"  [{name} attempt {i+1}/{attempts}: no file written, retry]")
        except Exception as e:
            print(f"  [{name} attempt {i+1}/{attempts} failed: {e}]")
    return None


def _default_name(source: str, cfg: Config) -> str:
    return {
        "landsat": f"lst_landsat_{cfg.aoi_name}.tif",
        "sentinel": f"s2_lulc_{cfg.aoi_name}.tif",
        "era5": f"era5_{cfg.aoi_name}.tif",
        "ghsl": f"ghsl_{cfg.aoi_name}.tif",
        "ghsl_h": f"ghsl_h_{cfg.aoi_name}.tif",
        "terrain": f"terrain_{cfg.aoi_name}.tif",
        "ecostress": f"ecostress_{cfg.aoi_name}.tif",
    }[source]


def ecostress_validation(cfg: Config) -> dict | None:
    """Best-effort cross-sensor LST validation: export ECOSTRESS, align to the
    Landsat grid, return agreement stats vs the Landsat composite. Returns None
    on any failure or insufficient ISS coverage (never raises)."""
    from . import ecostress
    ls = cfg.path("raw") / _default_name("landsat", cfg)
    if not _valid(ls):
        return None
    eco = _export_with_retry("ecostress", ecostress.export, cfg, attempts=2)
    if not eco:
        cached = cfg.path("raw") / _default_name("ecostress", cfg)
        eco = str(cached) if _valid(cached) else None
    if not eco:
        print("  [ecostress: no scenes / export failed -> validation skipped]")
        return None
    try:
        bands = align_stack({"landsat": str(ls), "ecostress": eco}, ref="landsat")
        stats = ecostress.agreement(bands["LST"], bands["LST_ECO"])
        if stats:
            print(f"  [ecostress cross-sensor: r={stats['pearson_r']}, "
                  f"MAE={stats['mae_C']}C, bias={stats['bias_C']}C]")
        return stats
    except Exception as e:
        print(f"  [ecostress validation skipped: {e}]")
        return None
