"""Sentinel-2 surface reflectance -> LULC + spectral indices, plus ESA WorldCover.

Outputs a multiband composite: B2,B3,B4,B8,B11 + NDVI,NDBI,NDWI and a WorldCover
land-cover class band (used to map interventions to eligible pixels).
"""
from __future__ import annotations
from pathlib import Path

from ..config import Config
from .gee_init import init_ee, aoi_geometry, safe_scale


def _mask_s2_clouds(img):
    qa = img.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus = qa.bitwiseAnd(1 << 11).eq(0)
    return img.updateMask(cloud).updateMask(cirrus).divide(10000)


def s2_composite(cfg: Config):
    """Cloud-free median Sentinel-2 SR with NDVI/NDBI/NDWI + WorldCover class."""
    ee = init_ee(cfg.gee_project)
    aoi = aoi_geometry(cfg)
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(cfg.start, cfg.end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .map(_mask_s2_clouds)
        .median()
        .clip(aoi)
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")   # built-up
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")    # water
    # Real broadband surface albedo from S2 narrowbands
    # (Bonafoni & Sekertekin 2020 narrow-to-broadband coefficients).
    albedo = s2.expression(
        "0.2266*B2 + 0.1236*B4 + 0.1573*B8 + 0.3417*B11 + 0.1170*B12",
        {"B2": s2.select("B2"), "B4": s2.select("B4"), "B8": s2.select("B8"),
         "B11": s2.select("B11"), "B12": s2.select("B12")},
    ).rename("ALBEDO")
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200").first()
        .rename("LULC").clip(aoi)
    )
    return (
        s2.select(["B2", "B3", "B4", "B8", "B11", "B12"])
        .addBands([ndvi, ndbi, ndwi, albedo, worldcover])
    )


def export(cfg: Config, out: Path | None = None) -> Path:
    import geemap
    out = out or (cfg.path("raw") / f"s2_lulc_{cfg.aoi_name}.tif")
    geemap.ee_export_image(
        s2_composite(cfg), filename=str(out),
        scale=safe_scale(cfg, 11),         # 11 bands -> coarsens on big AOIs
        region=aoi_geometry(cfg), file_per_band=False,
    )
    return out
