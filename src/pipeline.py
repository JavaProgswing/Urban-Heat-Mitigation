"""Reusable end-to-end analysis — shared by the CLI and the dashboard.

run_analysis(cfg, source, model) does the full chain (load drivers -> train ->
scenarios -> optimize) and returns everything needed to render maps/plans.
Keeping it here means the CLI and UI can never drift apart.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .config import Config
from .features import drivers, synthetic
from .models.train import train_xgb, train_pinn


@dataclass
class Analysis:
    cfg: Config
    source: str
    model: str
    stack: dict          # {name: 2-D ndarray} aligned driver rasters + LST
    df: Any              # per-pixel DataFrame (drivers + LST + hotspot flag)
    shape: tuple         # grid shape for re-gridding predictions
    res: Any             # TrainResult (model, metrics, cv, importance, eval)
    scenarios: dict      # {strategy: ScenarioResult}
    plan: Any            # optimized intervention DataFrame
    atmosphere: dict     # scene-mean ERA5 forcing (air temp, RH, wind)
    lst_validation: dict | None = None   # ECOSTRESS cross-sensor agreement


def _atmosphere(df) -> dict:
    """Scene-mean atmospheric forcing from ERA5 (air temp / humidity / wind).

    PS requires quantifying atmospheric conditions. At the satellite scale ERA5
    (~11 km) is near-uniform across a city, so it sets the LST BASELINE but does
    not drive intra-city hotspot VARIATION (which is surface-driven) — that is
    why it is reported as scene context here rather than used as a per-pixel
    model feature (including it only overfits the scene; see drivers.py)."""
    out = {}
    for col, key in (("AIR_T", "air_temp_C"), ("RH", "humidity_pct"),
                     ("WIND", "wind_m_s")):
        if col in df:
            out[key] = round(float(df[col].mean()), 2)
    return out


def load_drivers(cfg: Config, source: str) -> dict:
    """Driver stack from synthetic generator or live Earth Engine."""
    if source == "synthetic":
        return synthetic.make_grid(n=128)
    from .data.align import build_driver_stack    # lazy: needs GEE deps
    return build_driver_stack(cfg)


def run_analysis(cfg: Config, source: str = "synthetic",
                 model: str = "xgb", cv: bool = False,
                 hotspot_pct: float = 90.0, budget_frac: float = 0.3) -> Analysis:
    from .scenarios import cooling

    stack = load_drivers(cfg, source)
    shape = stack[drivers.TARGET_COL].shape
    df = drivers.stack_to_frame(stack)
    if len(df) < 200:
        raise RuntimeError(
            f"Only {len(df)} valid pixels after alignment — the AOI likely has "
            "little/no cloud-free satellite coverage in this date window. Widen "
            "the dates or move the AOI.")
    if "POP" not in df:
        df["POP"] = 1.0
    df = drivers.hotspots(df, pct=hotspot_pct)

    res = (train_pinn(df) if model == "pinn"
           else train_xgb(df, cv=cv, physics=True))
    atmo = _atmosphere(df)
    # humid climates suppress evaporative cooling (greening/water) — scale by RH.
    evap = cooling.humidity_evap_factor(atmo.get("humidity_pct"))
    scen = cooling.simulate_all(res.model, res.feature_names, df, cfg.scenarios,
                                lst_per_albedo=cfg.lst_per_albedo, shape=shape,
                                lst_per_ndwi=cfg.lst_per_ndwi,
                                lst_per_ndvi=cfg.lst_per_ndvi,
                                max_cooling_C=cfg.max_cooling_C, evap_factor=evap)
    plan = cooling.optimize(scen, df, budget_frac=budget_frac)

    lst_val = None
    if source == "gee" and cfg.validate_ecostress:
        try:
            from .data.align import ecostress_validation
            lst_val = ecostress_validation(cfg)
        except Exception as e:
            print(f"  [ecostress validation error: {e}]")

    return Analysis(cfg=cfg, source=source, model=model, stack=stack, df=df,
                    shape=shape, res=res, scenarios=scen, plan=plan,
                    atmosphere=atmo, lst_validation=lst_val)
