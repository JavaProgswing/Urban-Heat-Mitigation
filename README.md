# Urban Heat Mitigation AI/ML

Geospatial, physics-informed AI/ML system to (1) map urban heat hotspots, (2) quantify drivers of urban heating, (3) model land surface temperature (LST) dynamics, and (4) generate + optimize cooling interventions.

## Pipeline

```
acquire datasets ──> build drivers ──> train physics-informed model ──> simulate cooling scenarios ──> optimize + map
   (GEE/OSM)           (NDVI/NDBI...)       (monotonic XGBoost)          (cool roofs, greening...)
```

## Objectives → Modules

| Objective | Module |
|-----------|--------|
| Identify heat hotspots | `src/features/drivers.py` (LST), `src/viz/maps.py` |
| Analyze drivers of heating | `src/features/drivers.py`, `src/features/indices.py` |
| Model heat dynamics (physics-informed ML) | `src/models/train.py` |
| Generate + optimize cooling scenarios | `src/scenarios/cooling.py` |

## Datasets used

| Category | Source | Module | Variable |
|----------|--------|--------|----------|
| LST | Landsat 8 (Collection 2 L2, ST_B10) | `src/data/landsat.py` | Land surface temp |
| LST (high-res) | ECOSTRESS (ECO_L2T_LSTE, ~70 m) | `src/data/ecostress.py` | **cross-sensor validation** of the Landsat LST (`--validate` / `validate.ecostress`) |
| LULC | Sentinel-2 SR / ESA WorldCover | `src/data/sentinel.py` | Land use/land cover, NDVI/NDBI/NDWI |
| Meteorology | ERA5-Land + CPCB | `src/data/era5.py` | Air temp, humidity, wind, RH |
| Intervention geometry | OpenStreetMap (osmnx) | `src/data/osm.py`, `src/features/parcels.py` | Buildings, roads, parks/open land, water |
| Settlement / built-up | GHSL (built-up, pop) | `src/data/ghsl.py` | Built fraction, population |

## Setup

The production stack is live satellite data plus physics-constrained XGBoost:

```powershell
cd C:\Users\yashasvi\Documents\Python\urban-heat-ai
python -m pip install -r requirements.txt
```

Real-data (one-time, **do on Windows** — browser opens for login):
```powershell
# Set gee.project in config.yaml first, then authenticate + verify:
python scripts\gee_auth.py
```

## Run (Windows)

**Interactive UI (recommended)** — pick any city/region and dates, then run the live analysis:
```powershell
streamlit run dashboard\app.py
```

**CLI** — any place, any dates:
```powershell
python scripts\run_pipeline.py                                  # AOI in config.yaml
python scripts\run_pipeline.py --city "Mumbai"                  # any city/state by name
python scripts\run_pipeline.py --city "Bengaluru" --start 2024-03-01 --end 2024-06-01
python scripts\run_pipeline.py --bbox 72.8,18.9,73.0,19.2       # manual box
```

**Verify the data + predictions actually used:**
```powershell
python scripts\inspect_data.py     # cached raster stats + driver table + predictions
```

**Tests:**
```powershell
python -m pytest -q
```

Outputs land in `outputs/`: `lst_hotspots.png`, `driver_importance.png`, `validation_scatter.png`, `scenario_cooling.png`, `lst_map.html` (open in browser), `intervention_plan.csv`, `summary.json`, saved model. The dashboard additionally exposes an OSM parcel plan as downloadable GeoJSON.

## Production model

XGBoost uses **hard monotonic constraints** so increased vegetation, water, or albedo cannot produce physically reversed intervention estimates. It is deterministic, fast, and the only supported model path.

## Accuracy

Reported on real Indian-city LST (New Delhi, Bengaluru):

| Metric | Value | Meaning |
|--------|-------|---------|
| **Honest R²** | **0.43 – 0.58** | leakage-free skill on a held-out spatial quadrant (2×2 spatial CV) — the number to quote |
| In-scene R² | 0.70 – 0.81 | random-pixel split; optimistic (neighbour leakage) |
| MAE | 0.84 – 1.03 °C | typical absolute error |

The honest R² is what matters; the in-scene↔honest gap is the inherent ceiling of single-scene LST, verified un-closeable by tuning. What lifted it (all measured on cached tiles):
- **Dropped ERA5 meteorology from the model** — at city scale it's a near-constant position proxy that overfits (honest +0.06 to +0.15).
- **Multi-scale neighbourhood context** (`_N` 210 m + `_NC` 630 m) — matches LST's true spatial footprint (essential: removing it drops honest to 0.19).
- **2×2 quadrant CV** for an unbiased honest estimate (vs a single unlucky corner).
- **Landsat 8 + 9 merge** and **multi-year seasonal LST composite** (`time.lst_years`) — denser, cleaner target.

Physics-informed drivers and monotonic constraints keep interventions physical.

## Status

Complete + tested end-to-end. The pipeline exports and reproject-matches all datasets to the Landsat grid (`src/data/align.py`), auto-scales exports to the GEE size limit, and is resilient to a failed side source. It needs Earth Engine authentication (`python scripts/gee_auth.py`), a GEE project id, and an AOI. One-page PDF/PNG reports are written to `outputs/`.
