"""Quantify the context-refresh fix: scenario cooling with vs without recomputing
_N features under intervention. 'old' = shape=None (stale context), 'new' = fixed.

    python scripts/measure_scenarios.py lucknow
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.align import align_stack, derive_drivers
from src.features import drivers
from src.models.train import train_xgb
from src.scenarios import cooling


def cached_paths(aoi):
    raw = ROOT / "data" / "raw"
    n = {"landsat": f"lst_landsat_{aoi}.tif", "sentinel": f"s2_lulc_{aoi}.tif",
         "era5": f"era5_{aoi}.tif", "ghsl": f"ghsl_{aoi}.tif",
         "terrain": f"terrain_{aoi}.tif"}
    return {k: str(raw / v) for k, v in n.items() if (raw / v).exists()}


def main(aoi="lucknow"):
    stack = derive_drivers(align_stack(cached_paths(aoi), ref="landsat"))
    shape = stack["LST"].shape
    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0
    df = drivers.hotspots(df)
    res = train_xgb(df, cv=False, physics=True)
    cfg = load_config()

    old = cooling.simulate_all(res.model, res.feature_names, df, cfg.scenarios,
                               lst_per_albedo=cfg.lst_per_albedo, shape=None)
    new = cooling.simulate_all(res.model, res.feature_names, df, cfg.scenarios,
                               lst_per_albedo=cfg.lst_per_albedo, shape=shape)

    print(f"\n=== {aoi}  mean cooling (deg C) ===")
    print(f"  {'strategy':18s} {'old(stale _N)':>14s} {'new(refresh)':>14s} {'delta':>8s}")
    for k in old:
        o, n = old[k].mean_cooling, new[k].mean_cooling
        print(f"  {k:18s} {o:14.2f} {n:14.2f} {n - o:+8.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lucknow")
