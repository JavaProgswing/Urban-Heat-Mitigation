"""Aggregate per-pixel results into named neighbourhoods.

Turns the 30 m LST/plan rasters into a small table of zones — mean surface temp,
heat-risk index, population exposed, dominant cooling strategy — for the
neighbourhood heat map + prioritisation panels.

Zones come from real OpenStreetMap localities (place=suburb/neighbourhood/...)
when available, so they carry actual names (e.g. "Hauz Khas"); otherwise a 3x3
compass grid ("North", "Downtown", ...) is used so the feature always works.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Zone:
    name: str
    lon: float
    lat: float


def pixel_lonlat(bbox, shape):
    """Per-pixel centre lon/lat arrays for the aligned grid (row 0 = north)."""
    minx, miny, maxx, maxy = bbox
    h, w = shape
    lons = minx + (np.arange(w) + 0.5) / w * (maxx - minx)
    lats = maxy - (np.arange(h) + 0.5) / h * (maxy - miny)
    return lons, lats


def _grid_zones(bbox, n=3) -> list[Zone]:
    """Fallback: n x n compass-named zones spanning the AOI."""
    minx, miny, maxx, maxy = bbox
    rows = ["North", "", "South"] if n == 3 else [f"R{i}" for i in range(n)]
    cols = ["West", "Central", "East"] if n == 3 else [f"C{j}" for j in range(n)]
    out = []
    for i in range(n):
        for j in range(n):
            lon = minx + (j + 0.5) / n * (maxx - minx)
            lat = maxy - (i + 0.5) / n * (maxy - miny)
            label = " ".join(p for p in (rows[i], cols[j]) if p) or "Central"
            label = "Downtown" if (i == n // 2 and j == n // 2) else label
            out.append(Zone(label, lon, lat))
    return out


def osm_localities(bbox, limit=12) -> list[Zone]:
    """Named localities inside the AOI from OpenStreetMap (best-effort)."""
    import osmnx as ox
    minx, miny, maxx, maxy = bbox
    tags = {"place": ["suburb", "neighbourhood", "quarter", "city_district",
                      "village", "town", "hamlet", "locality"]}
    gdf = ox.features_from_bbox(maxy, miny, maxx, minx, tags=tags)
    if gdf is None or gdf.empty or "name" not in gdf.columns:
        return []
    gdf = gdf[gdf["name"].notna()].copy()
    pts = gdf.geometry.representative_point()
    gdf["lon"], gdf["lat"] = pts.x.values, pts.y.values
    # keep those whose centre falls inside the AOI; de-dup by name
    inside = ((gdf["lon"] >= minx) & (gdf["lon"] <= maxx) &
              (gdf["lat"] >= miny) & (gdf["lat"] <= maxy))
    gdf = gdf[inside].drop_duplicates("name")
    zones = [Zone(str(r["name"]), float(r["lon"]), float(r["lat"]))
             for _, r in gdf.iterrows()]
    # cap to the busiest few so labels don't overcrowd the map
    return zones[:limit] if zones else []


def assign(df, bbox, shape, zones: list[Zone]) -> np.ndarray:
    """Nearest-zone index for every pixel row in df (by lon/lat)."""
    lons, lats = pixel_lonlat(bbox, shape)
    plon = lons[df["col"].to_numpy()]
    plat = lats[df["row"].to_numpy()]
    zlon = np.array([z.lon for z in zones])
    zlat = np.array([z.lat for z in zones])
    # squared euclidean in degrees (AOI is small -> fine); nearest zone centroid
    d = (plon[:, None] - zlon[None, :]) ** 2 + (plat[:, None] - zlat[None, :]) ** 2
    return d.argmin(axis=1)


def neighborhood_table(df, bbox, shape, plan=None, scen_names=None):
    """Per-zone summary DataFrame + the per-pixel zone assignment.

    Columns: zone, mean_lst, heat_risk (0-100), pop_exposed, area_cells,
    top_strategy, priority. Sorted hottest-first.
    """
    try:
        zones = osm_localities(bbox)
        source = "osm"
    except Exception:
        zones, source = [], "grid"
    if len(zones) < 2:
        zones, source = _grid_zones(bbox), "grid"

    zi = assign(df, bbox, shape, zones)
    lst = df["LST"].to_numpy("float32")
    pop = df["POP"].to_numpy("float32") if "POP" in df else np.ones(len(df), "float32")

    # plan: best strategy per pixel (aligned to df order via row/col)
    strat = None
    if plan is not None and len(plan):
        key = df["row"].to_numpy() * (shape[1] + 1) + df["col"].to_numpy()
        pk = plan["row"].to_numpy() * (shape[1] + 1) + plan["col"].to_numpy()
        lut = dict(zip(pk, plan["best_strategy"]))
        strat = np.array([lut.get(k, None) for k in key], dtype=object)

    rows = []
    for i, z in enumerate(zones):
        m = zi == i
        if m.sum() < max(5, len(df) // 500):       # skip empty / tiny zones
            continue
        top = None
        if strat is not None:
            s = pd.Series(strat[m]).dropna()
            top = s.mode().iat[0] if len(s) else None
        rows.append({
            "zone": z.name,
            "row": int(np.round(df["row"].to_numpy()[m].mean())),
            "col": int(np.round(df["col"].to_numpy()[m].mean())),
            "mean_lst": round(float(lst[m].mean()), 1),
            "pop_exposed": int(pop[m].sum()),
            "area_cells": int(m.sum()),
            "top_strategy": top,
        })
    if not rows:
        return pd.DataFrame(), zi, source

    out = pd.DataFrame(rows)
    lo, hi = out["mean_lst"].min(), out["mean_lst"].max()
    out["heat_risk"] = (((out["mean_lst"] - lo) / (hi - lo + 1e-9)) * 100
                        ).round().astype(int)
    # priority = hot AND populated (top third by heat x log-exposure)
    rank = out["heat_risk"] * np.log1p(out["pop_exposed"].clip(lower=0))
    out["priority"] = rank >= rank.quantile(2 / 3)
    out = out.sort_values("mean_lst", ascending=False).reset_index(drop=True)
    return out, zi, source
