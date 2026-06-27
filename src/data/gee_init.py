"""Shared Google Earth Engine init + AOI geometry helper."""
from __future__ import annotations
import math
from functools import lru_cache

from ..config import Config

# geemap's ee_export_image uses the getPixels endpoint, capped at 48 MiB per
# request (50331648 bytes). Stay safely under it.
_GETPIXELS_LIMIT = 45_000_000


@lru_cache(maxsize=1)
def init_ee(project: str):
    """Authenticate-once / initialize Earth Engine. Call before any ee op."""
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def aoi_geometry(cfg: Config):
    ee = init_ee(cfg.gee_project)
    minx, miny, maxx, maxy = cfg.bbox
    return ee.Geometry.Rectangle([minx, miny, maxx, maxy])


def aoi_area_m2(cfg: Config) -> float:
    minx, miny, maxx, maxy = cfg.bbox
    midlat = math.radians((miny + maxy) / 2.0)
    w = abs(maxx - minx) * 111_320.0 * math.cos(midlat)
    h = abs(maxy - miny) * 110_570.0
    return max(w * h, 1.0)


def safe_scale(cfg: Config, n_bands: int, target: int | None = None) -> int:
    """Pick the finest scale (m) whose export stays under the getPixels limit.

    bytes ~= (area / scale^2) * n_bands * 4. Never finer than `target` (the
    native/desired resolution); coarsens automatically for large AOIs so a big
    city box doesn't blow the 48 MiB cap. align_stack resamples to the LST grid
    afterwards, so a coarser side-layer is fine.
    """
    target = int(target or cfg.resolution_m)
    need = math.sqrt(aoi_area_m2(cfg) * n_bands * 4.0 / _GETPIXELS_LIMIT)
    scale = max(target, int(math.ceil(need / 5.0) * 5))   # round up to 5 m
    return scale
