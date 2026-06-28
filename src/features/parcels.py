"""Allocate coarse-grid cooling scenarios to real OSM intervention assets.

The thermal/model grid remains the evidence scale.  This module samples that
field at real-world assets and enforces semantic feasibility: roof strategies
only on buildings, pavement only on roads, and greening/water creation only on
mapped open land. Existing water is retained as context, not painted as a new
intervention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


ALLOWED_STRATEGIES = {
    "building": ("cool_roofs", "green_roofs", "high_albedo_paint"),
    "road": ("cool_pavements",),
    "open_land": ("urban_greening", "water_body"),
}

_UNNAMED_LABELS = {
    "building": "Building",
    "road": "Road segment",
    "open_land": "Open space",
    "water": "Water",
}


@dataclass
class ParcelPlan:
    actions: Any
    context: Any
    source: str = "OpenStreetMap"
    warning: str | None = None

    @property
    def priority(self):
        if self.actions is None or self.actions.empty:
            return self.actions
        return self.actions[self.actions["priority"]].copy()


def _empty_gdf():
    import geopandas as gpd
    return gpd.GeoDataFrame(
        columns=["asset_id", "asset_type", "name", "geometry"], crs=4326)


def _grid(values, df, shape, fill=np.nan, dtype="float32"):
    out = np.full(shape, fill, dtype=dtype)
    rows = df["row"].to_numpy("int32")
    cols = df["col"].to_numpy("int32")
    out[rows, cols] = np.asarray(values, dtype=dtype)
    return out


def _sample_indices(geometries, bbox, shape):
    """Raster row/column at each geometry's guaranteed-inside point."""
    points = geometries.representative_point()
    minx, miny, maxx, maxy = [float(v) for v in bbox]
    h, w = shape
    x = points.x.to_numpy("float64")
    y = points.y.to_numpy("float64")
    cols = np.floor((x - minx) / max(maxx - minx, 1e-12) * w).astype("int32")
    rows = np.floor((maxy - y) / max(maxy - miny, 1e-12) * h).astype("int32")
    valid = ((rows >= 0) & (rows < h) & (cols >= 0) & (cols < w))
    return rows, cols, valid


def _asset_measures(actions):
    """Attach footprint area/road length in metres for useful tooltips."""
    actions = actions.copy()
    actions["area_m2"] = np.nan
    actions["length_m"] = np.nan
    if actions.empty:
        return actions
    try:
        metric = actions.to_crs(actions.estimate_utm_crs())
        road = actions["asset_type"].eq("road").to_numpy()
        actions.loc[road, "length_m"] = metric.geometry.length.to_numpy()[road]
        actions.loc[~road, "area_m2"] = metric.geometry.area.to_numpy()[~road]
    except Exception:
        pass
    return actions


def location_references(assets, nearest_m: float = 500.0):
    """Attach nearby OSM context, coordinates, and a no-key Google Maps URL."""
    import geopandas as gpd

    assets = assets.copy().reset_index(drop=True)
    if assets.empty:
        for col in ("osm_name", "location_ref", "latitude", "longitude", "map_url"):
            if col not in assets:
                assets[col] = []
        return assets
    if "name" not in assets:
        assets["name"] = ""
    if "osm_name" not in assets:
        assets["osm_name"] = assets["name"]
    names = assets["osm_name"].fillna("").astype(str).str.strip()
    lower = names.str.lower()
    missing = (lower.isin({"", "nan", "none", "null", "<na>"}) |
               lower.str.startswith("unnamed "))

    points = gpd.GeoSeries(assets.geometry.representative_point(), crs=assets.crs)
    points = points.to_crs(4326)
    assets["longitude"] = points.x.to_numpy()
    assets["latitude"] = points.y.to_numpy()
    ref_name = names.copy()
    ref_name.loc[missing] = ""
    if missing.any() and (~missing).any():
        try:
            point_gdf = gpd.GeoDataFrame(
                {"asset_row": np.arange(len(assets)), "ref_name": names.to_numpy()},
                geometry=points, crs=4326).to_crs(assets.estimate_utm_crs())
            query = point_gdf.loc[missing.to_numpy(), ["asset_row", "geometry"]]
            refs = point_gdf.loc[~missing.to_numpy(), ["ref_name", "geometry"]]
            nearest = gpd.sjoin_nearest(
                query, refs, how="left", max_distance=nearest_m,
                distance_col="reference_distance_m")
            nearest = nearest.drop_duplicates("asset_row").set_index("asset_row")
            for idx, value in nearest["ref_name"].dropna().items():
                ref_name.iloc[int(idx)] = str(value)
        except Exception:
            pass

    fallback = assets["asset_type"].map(_UNNAMED_LABELS).fillna("Unnamed asset")
    coords = (assets["latitude"].map(lambda v: f"{v:.5f}") + ", " +
              assets["longitude"].map(lambda v: f"{v:.5f}"))
    display = names.copy()
    # Keep the asset name compact. Nearby context already has its own
    # ``location_ref`` field and repeating it here makes tables/tooltips noisy.
    display.loc[missing] = fallback.loc[missing]
    assets["name"] = display
    assets["location_ref"] = ref_name.mask(ref_name.eq(""), coords)
    assets["map_url"] = (
        "https://www.google.com/maps/search/?api=1&query=" +
        assets["latitude"].map(lambda v: f"{v:.6f}") + "%2C" +
        assets["longitude"].map(lambda v: f"{v:.6f}"))
    return assets


def display_names(assets):
    """Backward-compatible alias for active Streamlit session objects."""
    return location_references(assets)


def allocate_assets(assets, df, shape, scenarios: dict, bbox,
                    budget_frac: float = 0.30) -> ParcelPlan:
    """Map scenario cooling fields onto semantically compatible OSM assets."""
    import geopandas as gpd
    from shapely.geometry import box

    if assets is None or assets.empty:
        return ParcelPlan(_empty_gdf(), _empty_gdf(),
                          warning="No usable OSM intervention geometry was found.")

    assets = assets.to_crs(4326).copy()
    try:
        assets = gpd.clip(assets, box(*bbox), keep_geom_type=False)
    except Exception:
        pass
    assets = assets[~assets.geometry.is_empty & assets.geometry.notna()].copy()
    assets = location_references(assets)
    water = assets[assets["asset_type"].eq("water")].copy()
    candidates = assets[assets["asset_type"].isin(ALLOWED_STRATEGIES)].copy()
    if candidates.empty:
        return ParcelPlan(_empty_gdf(), water,
                          warning="OSM returned context but no actionable assets.")

    rows, cols, inside = _sample_indices(candidates.geometry, bbox, shape)
    candidates = candidates.loc[inside].copy().reset_index(drop=True)
    rows, cols = rows[inside], cols[inside]
    if candidates.empty:
        return ParcelPlan(_empty_gdf(), water,
                          warning="OSM assets did not overlap the model grid.")

    cooling_grids = {
        name: _grid(result.delta_lst, df, shape)
        for name, result in scenarios.items()
    }
    chosen = np.full(len(candidates), "none", dtype=object)
    cooling = np.zeros(len(candidates), dtype="float32")
    for kind, allowed in ALLOWED_STRATEGIES.items():
        idx = np.flatnonzero(candidates["asset_type"].eq(kind).to_numpy())
        names = [name for name in allowed if name in cooling_grids]
        if not len(idx) or not names:
            continue
        vals = np.column_stack([
            np.nan_to_num(cooling_grids[name][rows[idx], cols[idx]], nan=0.0)
            for name in names
        ])
        best = vals.argmax(axis=1)
        chosen[idx] = np.asarray(names, dtype=object)[best]
        cooling[idx] = vals[np.arange(len(idx)), best]

    candidates["strategy"] = chosen
    candidates["cooling_C"] = cooling
    candidates["row"], candidates["col"] = rows, cols
    candidates = candidates[candidates["cooling_C"] > 0].copy()
    if candidates.empty:
        return ParcelPlan(_empty_gdf(), water,
                          warning="No OSM assets intersected an eligible cooling cell.")

    temp_col = "predicted_LST" if "predicted_LST" in df else "LST"
    temp_grid = _grid(df[temp_col].to_numpy(), df, shape)
    pop_grid = _grid(df["POP"].to_numpy() if "POP" in df else np.ones(len(df)),
                     df, shape, fill=0.0)
    cr = candidates["row"].to_numpy("int32")
    cc = candidates["col"].to_numpy("int32")
    candidates["surface_temp_C"] = temp_grid[cr, cc]
    median_temp = float(np.nanmedian(df[temp_col]))
    candidates["heat_excess_C"] = np.maximum(
        candidates["surface_temp_C"].to_numpy() - median_temp, 0.0)
    candidates["population"] = np.nan_to_num(pop_grid[cr, cc], nan=0.0)
    candidates["priority_score"] = (
        candidates["cooling_C"].to_numpy() *
        (1.0 + candidates["heat_excess_C"].to_numpy()) *
        (1.0 + np.log1p(np.clip(candidates["population"].to_numpy(), 0, None)))
    )
    candidates = _asset_measures(candidates)
    candidates = candidates.sort_values("priority_score", ascending=False).reset_index(drop=True)
    k = max(1, int(math.ceil(float(budget_frac) * len(candidates))))
    candidates["priority"] = candidates.index < k
    return ParcelPlan(candidates, water)


def build_parcel_plan(cfg, df, shape, scenarios: dict, budget_frac: float = 0.30,
                      max_area_km2: float = 225.0) -> ParcelPlan:
    """Fetch/cache OSM assets and allocate the model plan onto them.

    Very large AOIs intentionally retain the raster fallback: city-wide OSM
    building downloads can overload Overpass and a browser cannot usefully draw
    hundreds of thousands of parcel polygons at once.
    """
    from ..data.gee_init import aoi_area_m2
    from ..data.osm import fetch_intervention_assets

    area_km2 = aoi_area_m2(cfg) / 1_000_000.0
    if area_km2 > max_area_km2:
        return ParcelPlan(
            _empty_gdf(), _empty_gdf(), warning=(
                f"Parcel view is limited to {max_area_km2:.0f} km²; this AOI is "
                f"about {area_km2:.0f} km². Draw a smaller area to load OSM assets."
            ))
    try:
        assets = fetch_intervention_assets(cfg)
        return allocate_assets(assets, df, shape, scenarios, cfg.bbox, budget_frac)
    except Exception as exc:
        return ParcelPlan(_empty_gdf(), _empty_gdf(), warning=(
            "OSM parcel download was unavailable; showing the honest raster "
            f"fallback instead ({type(exc).__name__}: {exc})."
        ))
