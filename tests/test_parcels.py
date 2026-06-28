"""Offline tests for parcel-aware intervention allocation (no OSM network)."""
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from src.features.parcels import allocate_assets, location_references


def test_unnamed_asset_uses_nearest_named_osm_reference():
    assets = gpd.GeoDataFrame([
        {"asset_id": "b1", "asset_type": "building", "name": None,
         "geometry": box(80.0000, 26.0000, 80.0001, 26.0001)},
        {"asset_id": "r1", "asset_type": "road", "name": "Station Road",
         "geometry": LineString([(80.0002, 26.0000), (80.0002, 26.0002)])},
    ], crs=4326)
    located = location_references(assets)
    building = located.loc[located["asset_type"].eq("building")].iloc[0]
    assert building["name"] == "Building"
    assert building["location_ref"] == "Station Road"
    assert "google.com/maps/search" in building["map_url"]


def test_assets_receive_only_semantically_feasible_strategies():
    shape = (4, 4)
    bbox = (0.0, 0.0, 4.0, 4.0)
    df = pd.DataFrame({
        "row": np.repeat(np.arange(4), 4),
        "col": np.tile(np.arange(4), 4),
        "LST": np.linspace(35, 30, 16),
        "predicted_LST": np.linspace(35, 30, 16),
        "POP": np.ones(16),
    })
    assets = gpd.GeoDataFrame([
        {"asset_id": "b1", "asset_type": "building", "name": np.nan,
         "geometry": box(0.1, 3.1, 0.9, 3.9)},
        {"asset_id": "r1", "asset_type": "road", "name": None,
         "geometry": LineString([(1.1, 3.5), (1.9, 3.5)])},
        {"asset_id": "p1", "asset_type": "open_land", "name": "Park",
         "geometry": box(2.1, 3.1, 2.9, 3.9)},
        {"asset_id": "w1", "asset_type": "water", "name": "Lake",
         "geometry": box(3.1, 3.1, 3.9, 3.9)},
    ], crs=4326)

    def values(**at):
        out = np.zeros(16, dtype="float32")
        for idx, value in at.items():
            out[int(idx)] = value
        return SimpleNamespace(delta_lst=out)

    scenarios = {
        "cool_roofs": values(**{"0": 4, "1": 9, "2": 9}),
        "cool_pavements": values(**{"0": 9, "1": 3, "2": 9}),
        "urban_greening": values(**{"0": 9, "1": 9, "2": 2}),
        "water_body": values(**{"2": 1}),
    }
    result = allocate_assets(assets, df, shape, scenarios, bbox, budget_frac=0.5)
    got = dict(zip(result.actions["asset_type"], result.actions["strategy"]))

    assert got == {
        "building": "cool_roofs",
        "road": "cool_pavements",
        "open_land": "urban_greening",
    }
    assert len(result.context) == 1
    assert result.context.iloc[0]["asset_type"] == "water"
    assert int(result.actions["priority"].sum()) == 2
    assert np.isfinite(result.actions["surface_temp_C"]).all()
    labels = dict(zip(result.actions["asset_type"], result.actions["name"]))
    assert labels["building"] == "Building"
    assert labels["road"] == "Road segment"
    assert not result.actions["name"].str.lower().eq("nan").any()
    assert result.actions["map_url"].str.startswith(
        "https://www.google.com/maps/search/?api=1&query=").all()
    assert result.actions["location_ref"].str.len().gt(0).all()
