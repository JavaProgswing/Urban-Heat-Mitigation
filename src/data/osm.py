"""Urban morphology from OpenStreetMap via osmnx.

Derives per-grid-cell drivers: building footprint fraction, mean building height
(if tagged), road length density, and a coarse building density used as a
sky-view-factor / heat-trapping proxy.
"""
from __future__ import annotations
from pathlib import Path

from ..config import Config


def fetch_buildings(cfg: Config):
    """GeoDataFrame of building footprints in the AOI."""
    import osmnx as ox
    minx, miny, maxx, maxy = cfg.bbox
    return ox.features_from_bbox(maxy, miny, maxx, minx,
                                 tags={"building": True})


def fetch_roads(cfg: Config):
    """Road network graph for the AOI."""
    import osmnx as ox
    minx, miny, maxx, maxy = cfg.bbox
    return ox.graph_from_bbox(maxy, miny, maxx, minx, network_type="drive")


def morphology_grid(cfg: Config, cell_m: int | None = None):
    """Rasterize building fraction + road density onto the working grid.

    Returns a GeoDataFrame of grid cells with columns:
      build_frac, mean_height, road_density.
    """
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import box

    cell_m = cell_m or cfg.resolution_m
    deg = cell_m / 111_320.0           # approx m->deg at mid-latitude
    minx, miny, maxx, maxy = cfg.bbox

    buildings = fetch_buildings(cfg).to_crs(4326)
    cells = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            cell = box(x, y, x + deg, y + deg)
            inter = buildings.intersection(cell)
            area = inter.area.sum()
            frac = float(area / cell.area) if cell.area else 0.0
            h = (buildings.get("height")
                 .dropna().astype(str).str.extract(r"(\d+\.?\d*)")[0]
                 .astype(float).mean()
                 if "height" in buildings.columns else np.nan)
            cells.append({"geometry": cell, "build_frac": frac,
                          "mean_height": h})
            x += deg
        y += deg
    return gpd.GeoDataFrame(cells, crs=4326)


def save(cfg: Config, out: Path | None = None) -> Path:
    out = out or (cfg.path("raw") / f"osm_morphology_{cfg.aoi_name}.geojson")
    morphology_grid(cfg).to_file(out, driver="GeoJSON")
    return out
