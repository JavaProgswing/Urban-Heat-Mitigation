# Architecture & Workings

Full reference: what each file does, how the system works, datasets, accuracy.

## 1. What the project performs

Four-stage decision-support system for urban heat:

1. **Identify hotspots** — build a land-surface-temperature (LST) grid, flag the hottest pixels (top decile).
2. **Quantify drivers** — train an ML model on LST vs physical/morphological drivers, then rank driver influence (SHAP / permutation importance).
3. **Model heat dynamics (physics-informed)** — the model is constrained so that physically-known relationships hold (more vegetation/albedo → cooler; more built-up → hotter). This keeps predictions sane when simulating interventions never seen in training.
4. **Generate + optimize cooling** — counterfactually apply each intervention (cool roofs, greening, water…), re-predict LST, measure ΔLST, then optimize *where* to act under a coverage budget weighted by population exposure.

## 2. Data flow

```
 data sources (GEE / OSM)                features              model              scenarios
 ─────────────────────────              ─────────             ───────            ──────────
 Landsat8 / ECOSTRESS  ── LST ─┐
 Sentinel-2  ── NDVI/NDBI/NDWI ─┤
 ERA5  ── AIR_T/RH/WIND ────────┼─► driver stack ─► train ─► LST model ─► apply edit ─► ΔLST ─► optimize
 OSM   ── build_frac/height ────┤   (per-pixel df)  (XGB/PINN)  R²≈0.87     (counterfactual)   (plan.csv)
 GHSL  ── built/POP ────────────┘
```

Offline mode swaps the data sources for `features/synthetic.py` (physically-plausible generated grid) so everything runs with no Earth Engine auth.

## 3. File-by-file

### Config & entry
| File | Role |
|------|------|
| `config.yaml` | AOI bbox, date window, GEE project, per-strategy physical effect priors, paths |
| `.env.example` | secrets template (GEE project id, CDS API key) → copy to `.env` |
| `src/config.py` | loads yaml+env into a `Config`; `.override(bbox,start,end,…)` for runtime AOI/date changes from the UI |
| `src/pipeline.py` | `run_analysis(cfg, source, model)` — the shared end-to-end function used by BOTH the CLI and the dashboard (single source of truth) |
| `src/geocode.py` | place name ('Mumbai', 'Karnataka') → AOI bbox (osmnx), clipped to a max-km box |
| `scripts/run_pipeline.py` | **CLI orchestrator** — wraps `run_analysis`, writes `outputs/`; flags `--city/--bbox/--start/--end/--source/--model` |
| `scripts/gee_auth.py` | Earth Engine auth (localhost flow, no gcloud); run on Windows so the browser opens |
| `scripts/inspect_data.py` | **verifier** — raster stats of downloaded tiles + per-pixel driver table + predictions vs observed |
| `dashboard/app.py` | Streamlit UI: pick region by name/bbox + dates + Demo/Live, runs analysis, shows heat map + drivers + scenarios + plan |
| `requirements-windows.txt` | Windows deps without torch (XGB path); `requirements.txt` = full incl. torch for WSL/PINN |
| `pytest.ini` | test config; disables a global web3 pytest plugin that breaks collection |

### Data acquisition (`src/data/`) — one module per requested dataset
| File | Dataset | Output |
|------|---------|--------|
| `gee_init.py` | shared Earth Engine init + AOI geometry | — |
| `landsat.py` | **Landsat 8** C2-L2 `ST_B10` | LST °C composite (cloud-masked median) |
| `ecostress.py` | **ECOSTRESS** L2T_LSTE | high-res (~70 m) LST °C; GEE or local GeoTIFF |
| `sentinel.py` | **Sentinel-2** SR + ESA WorldCover | bands + NDVI/NDBI/NDWI + **real broadband ALBEDO** (Bonafoni 2020) + LULC |
| `era5.py` | **ERA5-Land** (+ CPCB hook) | air temp, RH (Magnus from dewpoint), wind speed |
| `terrain.py` | **SRTM** (USGS/SRTMGL1_003) | ELEV — elevation driver (lapse rate) |
| `osm.py` | **OpenStreetMap** via osmnx | building footprint fraction, mean height, road density |
| `ghsl.py` | **GHSL** built-up + population | built fraction (impervious proxy), population (exposure) |

11 model drivers: NDVI, NDBI, NDWI, albedo (real S2), build_frac, ELEV, WATER_DIST (distance-to-water from NDWI), NDVI_STD (vegetation texture), AIR_T, RH, WIND. WATER_DIST + NDVI_STD are computed locally in `align.spatial_drivers` (no extra GEE).

### Features (`src/features/`)
| File | Role |
|------|------|
| `indices.py` | NDVI/NDBI/NDWI math; albedo (Liang broadband, or NDVI-proxy fallback) |
| `drivers.py` | **core schema** — `DRIVER_COLS`, `TARGET_COL`, `PHYSICS_SIGNS` (sign of ∂LST/∂driver), stack→per-pixel DataFrame, `split_xy`, `hotspots()` |
| `synthetic.py` | offline driver grid; LST generated from drivers via energy-balance-like relation + noise (gives the model a real signal to recover) |

### Models (`src/models/`)
| File | Role |
|------|------|
| `pinn.py` | **Physics-Informed NN** (PyTorch). Data loss = MSE; physics loss = autograd-gradient penalties enforcing `PHYSICS_SIGNS` pointwise. Normalizes X and y, ramps physics weight, seeded/deterministic. |
| `train.py` | `train_xgb` (monotonic-constrained gradient boosting), `train_pinn`, k-fold `cross_validate`, SHAP + permutation importance, rich metrics (MAE/RMSE/R²) |

### Scenarios & viz
| File | Role |
|------|------|
| `scenarios/cooling.py` | `apply_intervention` (edit eligible pixels' drivers), `simulate`/`simulate_all` (ΔLST = before − after), `optimize` (greedy best-per-pixel under budget, pop-weighted) |
| `viz/maps.py` | LST heatmap, driver-importance bar, scenario-cooling bar, predicted-vs-observed validation scatter, interactive folium Leaflet map |

## 4. Why "physics-informed" (the key idea)

A plain ML model learns *correlations* in the training data. When you simulate a cool roof you push albedo to values the model never saw — and an unconstrained model can predict **warming**, which is physically wrong (we observed exactly this). Two mechanisms prevent it, both driven by `PHYSICS_SIGNS`:

- **XGBoost** (`train_xgb`, default): `monotone_constraints` — a **hard** guarantee that ∂LST/∂albedo ≤ 0, ∂LST/∂NDVI ≤ 0, ∂LST/∂NDBI ≥ 0, etc. Deterministic, reliable for extrapolation. **Recommended for scenarios.**
- **PINN** (`train_pinn`): a **soft** autograd penalty on the same gradients. Fits LST as well (R²≈0.87) and gives smooth physical fields, but the soft penalty can still mildly violate monotonicity far out-of-distribution (e.g. water/NDWI). Advanced/experimental.

## 5. Accuracy (synthetic benchmark)

Default XGB, 128×128 synthetic grid, 5-fold CV:

| Metric | Value |
|--------|-------|
| Test R² | **0.866** |
| 5-fold CV R² | **0.866 ± 0.003** |
| Test MAE | **0.50 °C** |
| Test RMSE | **0.62 °C** |
| PINN test R² | 0.872 |

Top drivers recovered: `build_frac` (0.45) and `NDBI` (0.33) dominate heating — matches the planted physics. Scenario cooling (mean ΔLST): urban_greening 3.1 °C, green_roofs 2.1 °C, water_body 0.9 °C, cool_roofs 0.6 °C.

> These numbers are on *synthetic* data and validate the pipeline machinery, not a real city. Real accuracy depends on satellite/meteo data quality and will be lower; report it from held-out real LST once `--source gee` is wired.

## 6. Real-data path (wired)

`run_pipeline.py --source gee` → `data/align.py::build_driver_stack`:
1. Export each source to GeoTIFF via geemap (`landsat.export_lst`, `sentinel.export`, `era5.export`, `ghsl.export`).
2. `align_stack` opens every raster and `rioxarray.reproject_match`-es it onto the **Landsat LST grid** (the modeling reference) — handles differing resolutions/CRS (ERA5 ~11 km, Sentinel 10 m, GHSL 100 m all resampled to the LST grid).
3. `derive_drivers` adds `albedo` (NDVI proxy — Sentinel-2 lacks the exact Liang Landsat bands) and `build_frac` (GHSL built surface normalized by its 99th percentile).
4. Result → `drivers.stack_to_frame` → identical downstream path as synthetic.

Band layout per exported file is fixed in `align.BAND_LAYOUT`. The alignment core is pure rioxarray (no GEE) and is unit-tested offline in `tests/test_align.py`.

**To run for real:** `earthengine authenticate`, set `gee.project` in `config.yaml` (or `GEE_PROJECT_ID` in `.env`) and the AOI bbox, then `--source gee`. Report real accuracy from held-out LST.

## 7. Optional physical models

`SOLWEIG` (mean radiant temperature, human thermal comfort) and `InVEST Urban Cooling` can be layered in as additional drivers or for validation — not required for the core pipeline.
