# Architecture & Workings

Production architecture for the live satellite + XGBoost urban-heat system.

## 1. Decision workflow

1. **Identify hotspots** — create a seasonal Landsat LST grid and flag the hottest decile.
2. **Quantify drivers** — model LST from vegetation, built form, albedo, water, terrain, texture, and neighbourhood context; explain it with SHAP.
3. **Constrain the physics** — XGBoost monotonic constraints ensure vegetation, water, and albedo cannot reverse into heating effects during counterfactual simulation.
4. **Optimize interventions** — edit eligible drivers, estimate ΔLST, and rank actions using cooling × heat excess × population exposure.
5. **Place actions on real assets** — assign roof strategies to OSM buildings, pavement strategies to roads, and nature-based strategies to mapped open land.

## 2. Data flow

```
Landsat 8/9 LST ───────────────┐
Sentinel-2 + WorldCover ───────┤
GHSL built/pop/height ─────────┼─► aligned driver grid ─► monotonic XGBoost
SRTM terrain ──────────────────┤                              │
ERA5 scene context ────────────┘                              ▼
                                              SHAP + cooling counterfactuals
                                                             │
OSM buildings/roads/open land/water ─────────────────────────┤
                                                             ▼
                                              parcel GeoJSON + dashboard map
```

OSM does not increase thermal resolution. It supplies feasible placement geometry, nearby named references, coordinates, and map links after modeling.

## 3. Main modules

### Entry and orchestration

| File | Role |
|------|------|
| `config.yaml` | AOI, date window, GEE project, physical scenario priors, paths |
| `src/pipeline.py` | Single production chain: live acquisition → XGBoost → scenarios → optimization |
| `scripts/run_pipeline.py` | CLI for a configured AOI, city search, or manual bbox |
| `dashboard/app.py` | Streamlit interface and Leaflet parcel map |
| `scripts/inspect_data.py` | Inspect the cached live rasters and predictions |

### Data acquisition

| File | Dataset/output |
|------|----------------|
| `src/data/landsat.py` | Landsat 8/9 C2-L2 surface-temperature composite |
| `src/data/sentinel.py` | Sentinel-2 indices, albedo, and ESA WorldCover |
| `src/data/ghsl.py` | Built surface, population, and building height |
| `src/data/era5.py` | Air temperature, humidity, and wind scene context |
| `src/data/terrain.py` | SRTM elevation |
| `src/data/ecostress.py` | Optional independent LST validation |
| `src/data/osm.py` | Cached buildings, roads, open land, and water geometry |
| `src/data/align.py` | Reproject all raster sources to the Landsat reference grid |

### Features, model, and planning

| File | Role |
|------|------|
| `src/features/drivers.py` | Driver schema, physical signs, frame assembly, hotspots |
| `src/models/train.py` | XGBoost training, spatial validation, SHAP importance |
| `src/scenarios/cooling.py` | Counterfactual cooling and heat/exposure-aware ranking |
| `src/features/parcels.py` | Semantic OSM asset allocation and location references |
| `src/viz/maps.py` | Raster visualization helpers |

## 4. Model and validation

The only supported model is XGBoost with `monotone_constraints` derived from `PHYSICS_SIGNS`. It is early-stopped, regularized, and capped to a representative training sample for large AOIs while inference still covers every valid pixel.

Validation includes:

- random held-out pixels for in-scene R² and MAE;
- 2×2 spatial-quadrant holdout for leakage-resistant R²;
- optional Landsat-versus-ECOSTRESS agreement.

The spatial score is the headline metric because neighbouring raster pixels otherwise leak information across a random split.

## 5. Scenario and parcel constraints

- Cool roofs, green roofs, and high-albedo paint may be assigned only to buildings.
- Cool pavement may be assigned only to road geometry.
- Greening and water creation may be assigned only to mapped open land.
- Existing water is context rather than a newly proposed intervention.
- Unnamed assets use the nearest named OSM feature within 500 m; otherwise latitude/longitude is shown.
- Google Maps links use the public coordinate search URL and require no API key.
- AOIs above 225 km² retain a raster fallback to avoid unusably large Overpass and browser payloads.

## 6. Resolution boundary

The working grid is 30 m, but Landsat thermal measurements are natively coarser. Parcel geometry makes recommendations operationally readable; it does not claim building-scale temperature observations. Cooling and LST values shown on a parcel are samples from the supporting model grid.
