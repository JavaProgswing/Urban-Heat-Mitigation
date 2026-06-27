"""Turn a place name into an AOI bbox.

Works like a map search: cities/states/districts AND specific localities,
colleges, landmarks or addresses ('IIT Delhi', 'Koramangala Bengaluru', 'Hauz
Khas'). Used by the CLI (--city) and the dashboard.
"""
from __future__ import annotations
import math


def box_around(lat: float, lon: float, size_km: float):
    """Square bbox [minlon,minlat,maxlon,maxlat] of side size_km centred on a point."""
    half = size_km / 2.0
    dlat = half / 110.57
    dlon = half / (111.32 * max(math.cos(math.radians(lat)), 1e-3))
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def geocode_aoi(query: str, max_km: float = 40.0):
    """Return (bbox, display_name, full_bbox).

    bbox = a `max_km` square centred on the matched place — so the size slider
    fully controls the AOI, whether the match is a city, a locality, or a single
    landmark/college (point). full_bbox = the place's true polygon extent if it
    has one, else None.
    """
    import osmnx as ox

    try:                                   # places with a polygon footprint
        gdf = ox.geocoder.geocode_to_gdf(query)
        minx, miny, maxx, maxy = gdf.iloc[0].geometry.bounds
        full = [minx, miny, maxx, maxy]
        lat, lon = (miny + maxy) / 2, (minx + maxx) / 2
        name = str(gdf.iloc[0].get("display_name", query)).split(",")[0]
    except Exception:                      # point-only result (landmark/address)
        lat, lon = ox.geocode(query)       # (lat, lon); raises if not found
        full, name = None, query

    bbox = box_around(lat, lon, max_km)
    slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:40]
    return bbox, slug, full
