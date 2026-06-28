"""End-to-end pipeline runner (CLI).

  load drivers -> train physics-informed model -> quantify drivers -> map
  hotspots -> simulate cooling scenarios -> optimize placement -> write outputs.

Live data for the AOI in config.yaml:
    python scripts/run_pipeline.py
Live data for any place / dates (overrides config.yaml):
    python scripts/run_pipeline.py --city "Mumbai" --start 2024-04-01 --end 2024-06-15
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                       # noqa: E402
from src.features import drivers                         # noqa: E402
from src.pipeline import run_analysis                    # noqa: E402
from src.viz import maps                                 # noqa: E402
from src.insights import cost_effectiveness as _cost_effectiveness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--city", default=None,
                    help="place name -> AOI bbox (city/locality/college/landmark)")
    ap.add_argument("--size-km", type=float, default=25.0, dest="size_km",
                    help="AOI box size in km for --city (use 1-3 for a locality)")
    ap.add_argument("--bbox", default=None,
                    help="minlon,minlat,maxlon,maxlat (overrides --city)")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--validate", action="store_true",
                    help="cross-validate the LST against ECOSTRESS (Live only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.validate:
        cfg.raw.setdefault("validate", {})["ecostress"] = True

    # runtime AOI / date overrides
    bbox = name = None
    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]
        name = "custom_bbox"
    elif args.city:
        from src.geocode import geocode_aoi
        bbox, name, full = geocode_aoi(args.city, max_km=args.size_km)
        print(f"[geocode] {args.city} ({args.size_km}km) -> {name}  "
              f"bbox={[round(b,3) for b in bbox]}")
    cfg = cfg.override(bbox=bbox, name=name, start=args.start, end=args.end)
    try:
        _ = cfg.gee_project
    except ValueError as exc:
        ap.error(str(exc))
    out = cfg.path("outputs")

    print(f"[1/6] load live drivers (aoi={cfg.aoi_name}, "
          f"{cfg.start}..{cfg.end})")
    print("[gee] exporting + aligning Landsat LST, Sentinel-2, ERA5, GHSL ...")

    a = run_analysis(cfg, cv=True)
    res, df, shape, scen, plan = a.res, a.df, a.shape, a.scenarios, a.plan

    print(f"[2/6] train model (xgb)  metrics: {res.metrics}")
    if res.cv:
        print(f"      {res.cv['k']}-fold CV: "
              f"MAE {res.cv['mae_mean']:.3f}±{res.cv['mae_std']:.3f} C  "
              f"R2 {res.cv['r2_mean']:.3f}±{res.cv['r2_std']:.3f}")
    print(f"[3/6] driver importance: "
          f"{json.dumps({k: round(v,3) for k,v in res.importance.items()})}")

    print("[4/6] map hotspots + validation")
    lst_grid = maps.regrid(df, df[drivers.TARGET_COL].to_numpy(), shape)
    maps.save_heatmap(lst_grid, out / "lst_hotspots.png")
    maps.save_driver_bar(res.importance, out / "driver_importance.png")
    if res.eval:
        maps.save_validation_scatter(res.eval["y_true"], res.eval["y_pred"],
                                     out / "validation_scatter.png", res.metrics)
    try:
        maps.folium_lst(df, lst_grid, cfg.bbox, out / "lst_map.html")
    except Exception as e:
        print(f"      [skip folium] {e}")

    print("[5/6] simulate cooling scenarios")
    for nm, r in sorted(scen.items(), key=lambda kv: -kv[1].mean_cooling):
        print(f"      {nm:16s} mean cooling {r.mean_cooling:5.2f} C "
              f"over {r.pixels} px")
    maps.save_scenario_bar(scen, out / "scenario_cooling.png")

    print("[6/6] optimize placement")
    plan.to_csv(out / "intervention_plan.csv", index=False)
    try:
        import joblib
        joblib.dump(res.model, out / "model_xgb.joblib")
    except Exception as e:
        print(f"      [model save skipped] {e}")

    summary = {
        "aoi": cfg.aoi_name, "bbox": cfg.bbox,
        "dates": [cfg.start, cfg.end], "source": "gee", "model": "xgb",
        "atmosphere_era5": a.atmosphere,
        "ecostress_validation": a.lst_validation,
        "metrics": res.metrics, "cv": res.cv,
        "driver_importance": res.importance,
        "scenario_mean_cooling_C": {k: round(v.mean_cooling, 3)
                                    for k, v in scen.items()},
        "cost_effectiveness": _cost_effectiveness(scen).to_dict("records"),
        "top_strategy": plan["best_strategy"].mode().iat[0],
        "plan_rows": len(plan),
        "mean_planned_cooling_C": round(float(plan["cooling_C"].mean()), 3),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    try:                                             # one-page submission report
        from src.report import save_report
        png, pdf = save_report(a, cfg, out)
        print(f"      report -> {png.name}, {pdf.name}")
    except Exception as e:
        print(f"      [report skipped] {e}")

    print(f"\nDone. Outputs -> {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
