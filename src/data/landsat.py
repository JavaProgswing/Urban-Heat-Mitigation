"""Landsat 8 + 9 Collection-2 Level-2 land surface temperature (LST).

Band ST_B10 is scaled surface temp. Convert to deg C:
    LST = ST_B10 * 0.00341802 + 149.0 - 273.15
Two noise-reduction tricks make the LST target far cleaner (and thus more
predictable from stable drivers -> higher honest spatial R2):
  - merge Landsat 8 AND 9 (identical ST_B10 product, ~8-day combined revisit)
  - composite the SAME SEASON across several years (config time.lst_years):
    a single date window yields only a handful of clear passes, so its median
    still carries per-acquisition noise; stacking the season over N years gives
    many more observations and a robust climatological hot-season LST.
"""
from __future__ import annotations
from pathlib import Path

from ..config import Config
from .gee_init import init_ee, aoi_geometry, safe_scale

_LANDSAT_C2L2 = ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")


def _mask_l2_clouds(img):
    qa = img.select("QA_PIXEL")
    # bits 3 (cloud) and 4 (cloud shadow)
    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)
    return img.updateMask(cloud).updateMask(shadow)


def _season_ranges(start: str, end: str, years: int):
    """Same calendar window for each of the last `years` years (YYYY-shifted)."""
    sy, ey = int(start[:4]), int(end[:4])
    return [(f"{sy - k}{start[4:]}", f"{ey - k}{end[4:]}") for k in range(max(1, years))]


def _thermal_collection(ee, aoi, ranges):
    """Merged, cloud-masked ST_B10 (Landsat 8+9) over all the date ranges."""
    date_filter = ee.Filter.Or([ee.Filter.date(s, e) for s, e in ranges])

    def prep(asset):
        return (
            ee.ImageCollection(asset)
            .filterBounds(aoi)
            .filter(date_filter)
            .filter(ee.Filter.lt("CLOUD_COVER", 40))
            .map(_mask_l2_clouds)
            .select("ST_B10")
        )
    col = prep(_LANDSAT_C2L2[0])
    for asset in _LANDSAT_C2L2[1:]:
        col = col.merge(prep(asset))
    return col


def lst_composite(cfg: Config):
    """Return an ee.Image with band 'LST' (deg C) for the AOI/time window."""
    ee = init_ee(cfg.gee_project)
    aoi = aoi_geometry(cfg)
    ranges = _season_ranges(cfg.start, cfg.end, cfg.lst_years)
    col = _thermal_collection(ee, aoi, ranges)
    lst = (
        col.median()
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST")
        .clip(aoi)
    )
    return lst


def export_lst(cfg: Config, out: Path | None = None) -> Path:
    """Export LST GeoTIFF to local disk via geemap."""
    import geemap
    ee = init_ee(cfg.gee_project)
    out = out or (cfg.path("raw") / f"lst_landsat_{cfg.aoi_name}.tif")
    geemap.ee_export_image(
        lst_composite(cfg),
        filename=str(out),
        scale=safe_scale(cfg, 1),          # LST = reference grid; 1 band
        region=aoi_geometry(cfg),
        file_per_band=False,
    )
    return out
