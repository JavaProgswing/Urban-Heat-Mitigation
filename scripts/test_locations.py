"""End-to-end smoke + timing across every cached AOI (no GEE).

Runs the full chain — align -> train -> scenarios -> optimize — on each location's
cached tiles, times each phase, and validates the outputs (sane R2, no warming
scenarios, non-empty plan). Catches regressions + slow paths across cities with
different completeness (e.g. indore has only Landsat+Sentinel).

    python scripts/test_locations.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.align import align_stack, derive_drivers
from src.features import drivers
from src.models.train import train_xgb
from src.scenarios import cooling

BANDS = {"landsat": "lst_landsat_{a}.tif", "sentinel": "s2_lulc_{a}.tif",
         "era5": "era5_{a}.tif", "ghsl": "ghsl_{a}.tif",
         "terrain": "terrain_{a}.tif"}


def cached_aois():
    raw = ROOT / "data" / "raw"
    aois = {}
    for f in raw.glob("lst_landsat_*.tif"):
        a = f.name[len("lst_landsat_"):-4]
        paths = {k: str(raw / v.format(a=a)) for k, v in BANDS.items()
                 if (raw / v.format(a=a)).exists()}
        if "landsat" in paths and "sentinel" in paths:
            aois[a] = paths
    return aois


def run_one(aoi, paths, cfg):
    r = {"aoi": aoi}
    t = time.time()
    stack = derive_drivers(align_stack(paths, ref="landsat"))
    shape = stack["LST"].shape
    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0
    df = drivers.hotspots(df)
    r["rows"] = len(df)
    r["t_load"] = time.time() - t

    t = time.time()
    res = train_xgb(df, cv=False, physics=True)
    r["t_train"] = time.time() - t
    r["r2"] = res.metrics["r2"]
    r["honest"] = res.metrics.get("r2_spatial")
    r["nfeat"] = len(res.feature_names)

    t = time.time()
    scen = cooling.simulate_all(res.model, res.feature_names, df, cfg.scenarios,
                                lst_per_albedo=cfg.lst_per_albedo, shape=shape,
                                lst_per_ndwi=cfg.lst_per_ndwi)
    plan = cooling.optimize(scen, df, budget_frac=0.3)
    r["t_scen"] = time.time() - t
    r["min_cool"] = min(s.mean_cooling for s in scen.values())
    r["max_cool"] = max(s.mean_cooling for s in scen.values())
    r["water"] = scen["water_body"].mean_cooling if "water_body" in scen else None
    r["plan_rows"] = len(plan)
    r["top"] = plan["best_strategy"].mode().iat[0]
    r["total"] = r["t_load"] + r["t_train"] + r["t_scen"]

    # validations
    checks = {
        "r2_sane": -1 < r["r2"] <= 1,
        "no_warming": r["min_cool"] >= -1e-6,          # physics: cooling >= 0
        "plan_nonempty": r["plan_rows"] > 0,
        "time_ok": r["total"] < 90,                    # well-framed budget
    }
    r["checks"] = checks
    r["pass"] = all(checks.values())
    return r


def main():
    cfg = load_config()
    aois = cached_aois()
    print(f"testing {len(aois)} AOIs: {list(aois)}\n")
    rows = []
    for a, paths in sorted(aois.items()):
        miss = [k for k in BANDS if k not in paths]
        try:
            r = run_one(a, paths, cfg)
            rows.append(r)
            flag = "PASS" if r["pass"] else "FAIL " + str(
                [k for k, v in r["checks"].items() if not v])
            print(f"[{flag}] {a:11s} rows={r['rows']:7d} feat={r['nfeat']:2d} "
                  f"miss={miss or '-'}")
            print(f"        R2 in-scene={r['r2']:.3f} honest="
                  f"{r['honest']:.3f}  cooling {r['min_cool']:.2f}..{r['max_cool']:.2f}"
                  f"  water={r['water']:.2f}  top={r['top']}")
            print(f"        time  load={r['t_load']:.1f}s train={r['t_train']:.1f}s "
                  f"scen={r['t_scen']:.1f}s  TOTAL={r['total']:.1f}s\n")
        except Exception as e:
            import traceback
            print(f"[ERROR] {a}: {e}")
            traceback.print_exc()

    ok = sum(r["pass"] for r in rows)
    print(f"=== {ok}/{len(rows)} passed · "
          f"slowest {max((r['total'] for r in rows), default=0):.1f}s ===")


if __name__ == "__main__":
    main()
