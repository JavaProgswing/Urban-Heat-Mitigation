# Urban Heat Mitigation AI/ML

Geospatial, physics-informed AI/ML system to (1) map urban heat hotspots, (2) quantify drivers of urban heating, (3) model land surface temperature (LST) dynamics, and (4) generate + optimize cooling interventions.

## Pipeline

```
acquire datasets ──> build drivers ──> train physics-informed model ──> simulate cooling scenarios ──> optimize + map
   (GEE/OSM)           (NDVI/NDBI...)        (PINN)                      (cool roofs, greening...)
```

## Objectives → Modules

| Objective | Module |
|-----------|--------|
| Identify heat hotspots | `src/features/drivers.py` (LST), `src/viz/maps.py` |
| Analyze drivers of heating | `src/features/drivers.py`, `src/features/indices.py` |
| Model heat dynamics (physics-informed ML) | `src/models/pinn.py`, `src/models/train.py` |
| Generate + optimize cooling scenarios | `src/scenarios/cooling.py` |

## Datasets used

| Category | Source | Module | Variable |
|----------|--------|--------|----------|
| LST | Landsat 8 (Collection 2 L2, ST_B10) | `src/data/landsat.py` | Land surface temp |
| LST (high-res) | ECOSTRESS (ECO_L2T_LSTE, ~70 m) | `src/data/ecostress.py` | **cross-sensor validation** of the Landsat LST (`--validate` / `validate.ecostress`) |
| LULC | Sentinel-2 SR / ESA WorldCover | `src/data/sentinel.py` | Land use/land cover, NDVI/NDBI/NDWI |
| Meteorology | ERA5-Land + CPCB | `src/data/era5.py` | Air temp, humidity, wind, RH |
| Urban morphology | OpenStreetMap (osmnx) | `src/data/osm.py` | Building footprint, road density |
| Settlement / built-up | GHSL (built-up, pop) | `src/data/ghsl.py` | Built fraction, population |

Optional physical models: SOLWEIG (mean radiant temp), InVEST Urban Cooling.

## Setup

**Windows is the primary path** — the default XGB model needs no PyTorch, so everything (train, GEE export, plots, folium maps, dashboard) runs natively:

```powershell
cd C:\Users\yashasvi\Documents\Python\urban-heat-ai
python -m pip install -r requirements-windows.txt
python scripts\run_pipeline.py          # offline demo, no GEE
```

**WSL is optional** — only needed for the experimental PINN model (PyTorch's Windows wheel has a known `c10.dll` failure). The project is wired to `~/venv_climate`:
```bash
cd /mnt/c/Users/yashasvi/Documents/Python/urban-heat-ai
~/venv_climate/bin/python scripts/run_pipeline.py --model pinn
```

Real-data (one-time, **do on Windows** — browser opens for login):
```powershell
python scripts\gee_auth.py --project YOUR_GEE_PROJECT   # authenticate + test
copy .env.example .env                                  # set GEE_PROJECT_ID; edit AOI in config.yaml
```

## Run (Windows)

**Interactive UI (recommended)** — pick any city/region by name, choose dates + Demo/Live data, see heat maps + scenarios:
```powershell
streamlit run dashboard\app.py
```

**CLI** — any place, any dates:
```powershell
python scripts\run_pipeline.py                                   # demo (synthetic)
python scripts\run_pipeline.py --source gee                      # AOI in config.yaml
python scripts\run_pipeline.py --source gee --city "Mumbai"      # any city/state by name
python scripts\run_pipeline.py --source gee --city "Bengaluru" --start 2024-03-01 --end 2024-06-01
python scripts\run_pipeline.py --source gee --bbox 72.8,18.9,73.0,19.2   # manual box
```

**Verify the data + predictions actually used:**
```powershell
python scripts\inspect_data.py --source gee     # raster stats + driver table + predictions
```

**Tests:**
```powershell
python -m pytest -q
```

Outputs land in `outputs/`: `lst_hotspots.png`, `driver_importance.png`, `validation_scatter.png`, `scenario_cooling.png`, `lst_map.html` (open in browser), `intervention_plan.csv`, `summary.json`, saved model.

## Model choice

`xgb` (default, Windows) — **hard monotonic constraints**, reliable + deterministic, no torch. `pinn` (WSL only) — **soft-penalty** physics-informed NN, fits as well (R²≈0.87) but less reliable far out-of-distribution. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Accuracy

Reported on **real** Indian-city LST (New Delhi, Bengaluru), not synthetic:

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

26 physics-informed drivers; monotonic constraints keep interventions physical. 6 tests pass.

## Status

Complete + tested end-to-end. Real-data path: `--source gee` exports + reproject-matches all datasets to the Landsat grid (`src/data/align.py`), auto-scales exports to the GEE size limit, and is resilient to a failed source. Run for real needs only `earthengine authenticate` (`python scripts/gee_auth.py`) + GEE project id + AOI. One-page PDF/PNG report auto-written to `outputs/`. Diagnostics in `scripts/{eval_spatial,ablate_features,multicity_eval}.py`.
