# Model & data — how it works (and where OSM fits)

## Do we use OpenStreetMap (OSM)?

**Yes for place search, no for the heat model.**

| Use | OSM? | Where | Detail |
|-----|------|-------|--------|
| **Geocoding** the "Search a place" box (name → AOI bbox) | **Yes** | [`src/geocode.py`](../src/geocode.py) | `osmnx` → Nominatim. `geocode_to_gdf` for places with a polygon, `ox.geocode` for point landmarks (e.g. "IIT Delhi"). Returns a square bbox sized by the slider. |
| **Model drivers** (urban morphology features) | **No** | — | The temperature model does **not** read OSM. |
| **Intervention placement** | **Yes** | [`src/data/osm.py`](../src/data/osm.py), [`src/features/parcels.py`](../src/features/parcels.py) | Model fields are sampled onto real buildings, road segments and mapped open land. Unnamed assets receive nearby OSM references, coordinates, and map links. |

**Why GHSL remains the model morphology source:** GHSL is a consistent global raster (built-surface, population, building height) that aligns cleanly to the satellite grid. OSM coverage is uneven city-to-city, so it does not sharpen or train the LST field. OSM is used after modeling as a feasibility layer: cool roofs go to buildings, cool pavements to roads, and nature-based actions to mapped open land.

---

## What the model does

Predict **land surface temperature (LST)** at 30 m from physical/morphological drivers, then use that model to (a) attribute the heat, (b) simulate cooling interventions, and (c) produce a placement plan.

Pipeline (one chain, in [`src/pipeline.py`](../src/pipeline.py)):

```
GEE export → align to one 30 m grid → engineer features → train ML
          → SHAP drivers → simulate scenarios → optimize plan → validate
```

### 1. Data → one aligned grid
Every source is reprojected onto the **Landsat LST 30 m grid** (EPSG:4326) in [`src/data/align.py`](../src/data/align.py).

| Source | Asset | Gives |
|--------|-------|-------|
| Landsat 8 **+ 9** C2L2 | `LANDSAT/LC0{8,9}/C02/T1_L2` `ST_B10` | **LST target** (°C) |
| Sentinel-2 SR + ESA WorldCover | `COPERNICUS/S2_SR_HARMONIZED`, `ESA/WorldCover/v200` | NDVI, NDBI, NDWI, albedo, land-cover class |
| GHSL P2023A | `GHS_BUILT_S`, `GHS_POP`, `GHS_BUILT_H` | building fraction, population, building height |
| SRTM | terrain | elevation |
| ERA5-Land | `ECMWF/ERA5_LAND/HOURLY` | air temp, humidity, wind (**scene context only**) |
| ECOSTRESS | `NASA/ECOSTRESS/L2T_LSTE/002` | independent LST for **validation** (not a driver) |

**LST target is denoised**, not a single cloudy median: Landsat 8 + 9 merged and the **same hot season composited across 3 years** (`lst_years`) → many more clear passes → a stable climatological LST that's actually predictable.

### 2. Features (~25–29 per pixel) — [`src/features/drivers.py`](../src/features/drivers.py)
- **Base (8):** NDVI, NDBI, NDWI, albedo, build_frac, building height, elevation, distance-to-water.
- **Texture (4, `_STD`):** local 5-px std — surface heterogeneity.
- **Neighbourhood context (10, `_N` ≈210 m + `_NC` ≈630 m):** area-means of the 5 land drivers. Landsat thermal is ~100 m native, so a pixel's LST reflects its *surroundings* — **the single strongest predictors** (removing them collapses honest R² to ~0.19).
- **Land cover (7, one-hot):** WorldCover tree/grass/crop/built/bare/water/shrub.
- **ERA5 is excluded from the model** — at ~11 km it's near-constant across a city, so it only overfits (ablation: dropping it raised honest R² 0.42→0.56 on Delhi). Kept as reported scene context.

### 3. Model — physics-constrained ML — [`src/models/train.py`](../src/models/train.py)
- **XGBoost** gradient-boosted trees (`tree_method=hist`, early-stopped).
- **Monotonic physics constraints** from `PHYSICS_SIGNS`: NDVI/NDWI/albedo → cooler, NDBI/build_frac → hotter, etc. This is the "physics-informed" core — the model **cannot** predict warming from a cooling intervention, even when extrapolating to states it never saw.
- **SHAP** attribution → the "what drives heat" chart (raw features grouped into ~6 physical factors).

### 4. Validation — two numbers + an external check
- **In-scene R²** — random-pixel hold-out (optimistic; neighbours leak across the split).
- **Honest R²** — **2×2 spatial-quadrant cross-validation** (train 3 quadrants, test the 4th). No leakage → true skill on unseen ground. *This is the headline.* (Delhi ≈0.55, peri-urban Lucknow ≈0.27.)
- **ECOSTRESS cross-sensor** ([`src/data/ecostress.py`](../src/data/ecostress.py)): correlates the Landsat target with an independent ISS thermal sensor (r, MAE, bias) — confirms the *target* is real, not just that the model fits.

### 5. Cooling scenarios — counterfactual + physics — [`src/scenarios/cooling.py`](../src/scenarios/cooling.py)
For each of 6 strategies: edit the relevant driver on **eligible** pixels → re-predict LST → ΔLST = before − after. Made physical by:
- **Land-use-exclusive eligibility** (roofs in built core; greening/water on open land; water only near existing water).
- **Neighbourhood-context refresh** so a broad intervention's area effect reaches the model.
- **Physics floors via `max()` (not additive)** for confounded drivers — `lst_per_albedo=15`, `lst_per_ndvi=8`, `lst_per_ndwi=6` °C/unit.
- **Achievable-target scaling** — cooling ∝ (target − current), so a dark roof gains more than a bright one.
- **Humidity scaling** — evaporative strategies (greening/water) cool less in humid climates; albedo is unaffected.
- **Realism cap** `max_cooling_C=8` — bounds model over-extrapolation.

### 6. Optimizer — the plan
`optimize()`: per pixel pick the strategy that cools most, then rank by **cooling × heat excess × population exposure**. The parcel allocator applies the compatible scenario field to each OSM asset and independently flags the top fraction as **priority**. Output = **what** (strategy) + **where** (building/road/open-land geometry) + **how much** (°C), with raster CSV and parcel GeoJSON downloads.

---

**In one line:** satellite + reanalysis → one aligned 30 m stack → physically meaningful features → monotonic XGBoost (spatially cross-validated, ECOSTRESS-checked) → counterfactual, literature-anchored cooling scenarios → an OSM parcel plan with location references and map links.
