"""Terrain elevation from SRTM (USGS/SRTMGL1_003, 30 m).

Elevation is a physical LST driver (lapse rate: higher ground tends cooler) and
also encodes valleys/ridges that channel heat. Native 30 m EPSG:4326 -> clean
export onto the working grid.
"""
from __future__ import annotations
from pathlib import Path

from ..config import Config
from .gee_init import init_ee, aoi_geometry, safe_scale


def elevation(cfg: Config):
    """ee.Image with band ELEV (metres)."""
    ee = init_ee(cfg.gee_project)
    aoi = aoi_geometry(cfg)
    return (ee.Image("USGS/SRTMGL1_003")
            .select("elevation").rename("ELEV").clip(aoi))


def export(cfg: Config, out: Path | None = None) -> Path:
    import geemap
    out = out or (cfg.path("raw") / f"terrain_{cfg.aoi_name}.tif")
    geemap.ee_export_image(
        elevation(cfg), filename=str(out),
        scale=safe_scale(cfg, 1), crs=cfg.raw["project"]["crs"],
        region=aoi_geometry(cfg), file_per_band=False,
    )
    return out
