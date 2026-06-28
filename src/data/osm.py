"""OpenStreetMap acquisition for parcel-aware intervention placement.

OSM geometry is deliberately *not* used to sharpen Landsat temperature.  It is
the feasibility layer that turns a coarse thermal recommendation into actions
on real buildings, road segments, parks/open land, and existing water.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import Config


_OPEN_LANDUSE = {
    "allotments", "brownfield", "cemetery", "flowerbed", "grass",
    "greenfield", "meadow", "recreation_ground", "village_green",
}
_OPEN_LEISURE = {"garden", "park", "pitch", "playground"}
_OPEN_NATURAL = {"grassland", "heath", "scrub", "wood"}
_ROAD_TYPES = {
    "living_street", "motorway", "motorway_link", "primary", "primary_link",
    "residential", "secondary", "secondary_link", "service", "tertiary",
    "tertiary_link", "trunk", "trunk_link", "unclassified",
}


def _features_from_bbox(bbox, tags):
    """OSMnx 1.x/2.x compatible feature query."""
    import osmnx as ox

    try:  # OSMnx >= 2: (left, bottom, right, top), tags
        return ox.features_from_bbox(tuple(bbox), tags=tags)
    except TypeError:  # OSMnx 1.x: north, south, east, west, tags
        minx, miny, maxx, maxy = bbox
        return ox.features_from_bbox(maxy, miny, maxx, minx, tags=tags)


def fetch_buildings(cfg: Config):
    """GeoDataFrame of building footprints in the AOI."""
    gdf = _features_from_bbox(cfg.bbox, {"building": True})
    return gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


def fetch_roads(cfg: Config):
    """GeoDataFrame of driveable road centre-lines in the AOI."""
    gdf = _features_from_bbox(cfg.bbox, {"highway": list(_ROAD_TYPES)})
    return gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()


def _tag_value(row, key) -> str:
    value = row.get(key)
    if isinstance(value, (list, tuple, set)):
        value = next(iter(value), "")
    if value is None:
        return ""
    try:
        # OSM/GeoJSON round-trips commonly turn an absent name into float NaN.
        if value != value:  # NaN is the only ordinary value unequal to itself.
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "<na>"} else text


def _clean_names(gdf):
    """Normalize missing names after both Overpass and GeoJSON cache reads."""
    if "name" not in gdf:
        gdf["name"] = ""
        return gdf
    gdf = gdf.copy()
    gdf["name"] = gdf.apply(lambda row: _tag_value(row, "name"), axis=1)
    return gdf


def _asset_id(index) -> str:
    if isinstance(index, tuple):
        return "/".join(str(v) for v in index)
    return str(index)


def _classify(row) -> str | None:
    geom_type = row.geometry.geom_type
    polygon = geom_type in {"Polygon", "MultiPolygon"}
    line = geom_type in {"LineString", "MultiLineString"}
    if polygon and _tag_value(row, "building"):
        return "building"
    highway = _tag_value(row, "highway")
    if line and highway in _ROAD_TYPES:
        return "road"
    natural = _tag_value(row, "natural")
    if (polygon or line) and (
        natural == "water" or _tag_value(row, "water") or
        _tag_value(row, "waterway")
    ):
        return "water"
    if polygon and (
        _tag_value(row, "landuse") in _OPEN_LANDUSE or
        _tag_value(row, "leisure") in _OPEN_LEISURE or
        natural in _OPEN_NATURAL
    ):
        return "open_land"
    return None


def _cache_path(cfg: Config) -> Path:
    key = ",".join(f"{float(v):.5f}" for v in cfg.bbox)
    digest = hashlib.sha1(f"osm-assets-v2:{key}".encode()).hexdigest()[:12]
    return cfg.path("raw") / f"osm_assets_{digest}.geojson"


def fetch_intervention_assets(cfg: Config, reuse: bool = True):
    """Return normalized OSM assets, cached by exact AOI bounds.

    Columns are ``asset_id``, ``asset_type`` (building/road/open_land/water),
    ``name``, and ``geometry``.  The one-query union is materially faster and
    kinder to Overpass than separate city-scale calls for each asset class.
    """
    import geopandas as gpd
    from shapely.geometry import box

    cache = _cache_path(cfg)
    if reuse and cache.exists() and cache.stat().st_size > 256:
        try:
            cached = gpd.read_file(cache)
            if {"asset_id", "asset_type", "geometry"}.issubset(cached.columns):
                return _clean_names(cached.to_crs(4326))
        except Exception:
            pass

    tags = {
        "building": True,
        "highway": list(_ROAD_TYPES),
        "landuse": list(_OPEN_LANDUSE),
        "leisure": list(_OPEN_LEISURE),
        "natural": list(_OPEN_NATURAL | {"water"}),
        "water": True,
        "waterway": True,
    }
    raw = _features_from_bbox(cfg.bbox, tags)
    if raw is None or raw.empty:
        return gpd.GeoDataFrame(
            columns=["asset_id", "asset_type", "name", "geometry"], crs=4326)

    rows = []
    for idx, row in raw.iterrows():
        kind = _classify(row)
        geom = row.geometry
        if kind is None or geom is None or geom.is_empty:
            continue
        rows.append({
            "asset_id": _asset_id(idx),
            "asset_type": kind,
            "name": _tag_value(row, "name"),
            "geometry": geom,
        })
    assets = gpd.GeoDataFrame(rows, crs=raw.crs or 4326).to_crs(4326)
    if assets.empty:
        return assets

    assets = assets[assets.geometry.is_valid & ~assets.geometry.is_empty].copy()
    # Overpass returns complete ways that merely cross the query bbox. Clip them
    # before caching so long roads/rivers cannot leak outside the displayed AOI.
    assets = gpd.clip(assets, box(*cfg.bbox), keep_geom_type=False)
    assets = _clean_names(assets.drop_duplicates("asset_id").reset_index(drop=True))
    # A metre-level simplification keeps Leaflet responsive without changing the
    # planning scale. OSM's source geometries remain untouched in Overpass.
    try:
        metric = assets.to_crs(assets.estimate_utm_crs())
        metric.geometry = metric.geometry.simplify(1.0, preserve_topology=True)
        assets = metric.to_crs(4326)
    except Exception:
        pass
    cache.parent.mkdir(parents=True, exist_ok=True)
    assets.to_file(cache, driver="GeoJSON")
    return assets


def morphology_grid(cfg: Config, cell_m: int | None = None):
    """Rasterize building fraction onto a regular grid (legacy utility)."""
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import box

    cell_m = cell_m or cfg.resolution_m
    deg = cell_m / 111_320.0
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
