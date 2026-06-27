"""ECOSTRESS LST (ECO_L2T_LSTE, ~70 m thermal from the ISS).

Used as an INDEPENDENT cross-sensor validation of the Landsat LST target — if a
second satellite agrees on the surface temperature, the modelled target is
trustworthy (strengthens the "validated AI/ML model" outcome). NOT a model
driver. ISS coverage is sporadic, so compositing widens across years and the
whole step is best-effort (skips cleanly when no scenes intersect the AOI).
"""
from __future__ import annotations
from pathlib import Path

from ..config import Config
from .gee_init import init_ee, aoi_geometry, safe_scale

# ECOSTRESS L2T_LSTE: LST band stored in Kelvin * 0.02 (LP DAAC user guide).
_SCALE_K = 0.02
_ASSET = "NASA/ECOSTRESS/L2T_LSTE/002"


def lst_composite(cfg: Config, asset_id: str = _ASSET):
    """Median ECOSTRESS LST (deg C), composited over the season across years."""
    from .landsat import _season_ranges
    ee = init_ee(cfg.gee_project)
    aoi = aoi_geometry(cfg)
    ranges = _season_ranges(cfg.start, cfg.end, max(cfg.lst_years, 3))
    date_filter = ee.Filter.Or([ee.Filter.date(s, e) for s, e in ranges])
    col = ee.ImageCollection(asset_id).filterBounds(aoi).filter(date_filter)
    return (
        col.select("LST").median()
        .multiply(_SCALE_K).subtract(273.15)
        .rename("LST_ECO").clip(aoi)
    )


def export(cfg: Config, out: Path | None = None) -> Path:
    """Export an ECOSTRESS LST tile (~70 m) for cross-sensor validation."""
    import geemap
    out = out or (cfg.path("raw") / f"ecostress_{cfg.aoi_name}.tif")
    geemap.ee_export_image(
        lst_composite(cfg), filename=str(out),
        scale=safe_scale(cfg, 1, target=70), crs=cfg.raw["project"]["crs"],
        region=aoi_geometry(cfg), file_per_band=False,
    )
    return out


def agreement(landsat_lst, eco_lst) -> dict | None:
    """Cross-sensor agreement on overlapping valid pixels: r, MAE, bias.

    bias = mean(ECOSTRESS - Landsat). Returns None if too few overlapping pixels
    (sparse ISS coverage), so callers can degrade gracefully.
    """
    import numpy as np
    a = np.asarray(landsat_lst, "float32")
    b = np.asarray(eco_lst, "float32")
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 200:
        return None
    a, b = a[m], b[m]
    return {
        "n_pixels": int(a.size),
        "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 3),
        "mae_C": round(float(np.mean(np.abs(a - b))), 2),
        "bias_C": round(float(np.mean(b - a)), 2),
    }


def load_local(cfg: Config, filename: str | None = None):
    """Read a manually downloaded ECOSTRESS GeoTIFF (LP DAAC / AppEEARS)."""
    import rioxarray
    path = Path(filename or cfg.path("raw") / f"ecostress_{cfg.aoi_name}.tif")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download ECO_L2T_LSTE from LP DAAC/AppEEARS."
        )
    return rioxarray.open_rasterio(path, masked=True).squeeze()
