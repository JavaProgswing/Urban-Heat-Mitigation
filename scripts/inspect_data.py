"""Verify what data + predictions the system actually used.

Shows, for the current AOI:
  - each downloaded GeoTIFF: bands, shape, CRS, value ranges
  - the aligned per-pixel driver table (head + stats)
  - a sample of model predictions vs observed LST

    python scripts/inspect_data.py                 # synthetic
    python scripts/inspect_data.py --source gee     # the real tiles you exported
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                 # noqa: E402
from src.features import drivers, synthetic         # noqa: E402
from src.models.train import train_xgb              # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def show_rasters(cfg):
    import rioxarray
    raw = cfg.path("raw")
    tifs = sorted(raw.glob("*.tif"))
    if not tifs:
        print("  (no GeoTIFFs in data/raw — run --source gee first)")
        return
    print(f"\n=== Downloaded rasters in {raw} ===")
    for t in tifs:
        da = rioxarray.open_rasterio(t, masked=True)
        vals = np.asarray(da.values, dtype="float32")
        print(f"\n{t.name}")
        print(f"  bands={da.shape[0]}  HxW={da.shape[1]}x{da.shape[2]}  "
              f"crs={da.rio.crs}")
        for b in range(da.shape[0]):
            v = vals[b]
            v = v[np.isfinite(v)]
            if v.size:
                print(f"  band{b+1}: min={v.min():8.3f}  mean={v.mean():8.3f}  "
                      f"max={v.max():8.3f}")
        da.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "gee"], default="synthetic")
    ap.add_argument("--rows", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    print(f"AOI: {cfg.aoi_name}  bbox={cfg.bbox}  dates={cfg.start}..{cfg.end}")

    # Build the driver stack WITHOUT re-exporting: synthetic generator, or
    # align the GeoTIFFs already in data/raw (the exact tiles last used).
    if args.source == "gee":
        show_rasters(cfg)
        from src.data.align import align_stack, derive_drivers, _default_name
        paths = {}
        for name in ("landsat", "sentinel", "era5", "ghsl", "terrain"):
            p = cfg.path("raw") / _default_name(name, cfg)
            if p.exists():
                paths[name] = str(p)
        if "landsat" not in paths:
            print("\nNo cached Landsat tile. Run: python scripts/run_pipeline.py "
                  "--source gee")
            return
        stack = derive_drivers(align_stack(paths, ref="landsat"))
    else:
        stack = synthetic.make_grid(n=128)

    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0
    df = drivers.hotspots(df, pct=90.0)

    print(f"\n=== Per-pixel driver table ({len(df):,} pixels) ===")
    cols = [c for c in drivers.DRIVER_COLS if c in df] + [drivers.TARGET_COL]
    print(df[cols].head(args.rows).to_string(index=False))
    print("\nColumn stats:")
    print(df[cols].describe().loc[["min", "mean", "max"]].to_string())

    print("\n=== Training + predictions vs observed (random sample) ===")
    res = train_xgb(df, cv=False)
    X, y, names = drivers.split_xy(df)
    pred = res.model.predict(X)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(args.rows, len(y)), replace=False)
    smp = pd.DataFrame({"observed_LST": y[idx].round(2),
                        "predicted_LST": pred[idx].round(2),
                        "error": (pred[idx] - y[idx]).round(2)})
    print(smp.to_string(index=False))
    print(f"\nOverall: R2={res.metrics['r2']:.3f}  "
          f"MAE={res.metrics['mae']:.3f} C  features={names}")


if __name__ == "__main__":
    main()
