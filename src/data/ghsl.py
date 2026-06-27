"""Global Human Settlement Layer (GHSL): built-up surface + population.

Via GEE: JRC/GHSL/P2023A/GHS_BUILT_S and GHS_POP. Built fraction is an
impervious-surface proxy; population weights heat-risk prioritization.
"""
from __future__ import annotations
from pathlib import Path

from ..config import Config
from .gee_init import init_ee, aoi_geometry, safe_scale


def built_and_pop(cfg: Config, epoch: int = 2020):
    """ee.Image with bands BUILT (m2 built / cell) and POP (people / cell).

    GHSL P2023A are ImageCollections (one image per epoch) in Mollweide
    (ESRI:54009). Filter to the epoch; the export step handles the CRS/scale.
    """
    ee = init_ee(cfg.gee_project)
    aoi = aoi_geometry(cfg)

    built = (
        ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
        .filter(ee.Filter.eq("system:index", str(epoch))).first()
        .select("built_surface").rename("BUILT")
    )
    pop = (
        ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP")
        .filter(ee.Filter.eq("system:index", str(epoch))).first()
        .select("population_count").rename("POP")
    )
    # No explicit .reproject: forcing 100 m Mollweide -> 30 m here makes the
    # getPixels download fail. Clip only; the export crs + align step resample.
    return ee.Image.cat([built, pop]).clip(aoi)


def export(cfg: Config, out: Path | None = None, scale: int | None = None) -> Path:
    """Export at GHSL's native ~100 m (lighter download); align_stack upsamples
    it to the 30 m Landsat grid locally via reproject_match."""
    import geemap
    out = out or (cfg.path("raw") / f"ghsl_{cfg.aoi_name}.tif")
    scale = scale or safe_scale(cfg, 2, target=100)
    geemap.ee_export_image(
        built_and_pop(cfg), filename=str(out), scale=scale,
        crs=cfg.raw["project"]["crs"],
        region=aoi_geometry(cfg), file_per_band=False,
    )
    return out


def built_height(cfg: Config):
    """ee.Image with band BUILT_H = average gross building height (m) per cell.

    GHS_BUILT_H is the vertical morphology that BUILT/NDBI/LULC_built all miss
    (a 3-storey vs 30-storey built pixel behave very differently thermally).
    Exported as a SEPARATE source so a dataset/band change can never break the
    critical BUILT/POP export.
    """
    ee = init_ee(cfg.gee_project)
    aoi = aoi_geometry(cfg)
    h = (
        ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_H")
        .first()                                    # single 2018 epoch
        .select("built_height").rename("BUILT_H")
    )
    return h.clip(aoi)


def export_height(cfg: Config, out: Path | None = None, scale: int | None = None) -> Path:
    """Export GHSL building height (optional driver). Own tile; failure is
    non-fatal in build_driver_stack (BLD_H simply absent)."""
    import geemap
    out = out or (cfg.path("raw") / f"ghsl_h_{cfg.aoi_name}.tif")
    scale = scale or safe_scale(cfg, 1, target=100)
    geemap.ee_export_image(
        built_height(cfg), filename=str(out), scale=scale,
        crs=cfg.raw["project"]["crs"],
        region=aoi_geometry(cfg), file_per_band=False,
    )
    return out
