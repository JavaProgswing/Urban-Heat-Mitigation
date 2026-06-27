"""Measure LST-model accuracy on cached live tiles (no GEE re-export).

Aligns the locally cached GeoTIFFs for an AOI, derives drivers, trains the same
XGBoost the dashboard uses, and prints held-out + honest spatial-CV metrics.
Lets us compare accuracy before/after feature or training changes deterministically.

    python scripts/eval_accuracy.py new_delhi
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.align import align_stack, derive_drivers
from src.features import drivers
from src.models.train import train_xgb


def cached_paths(aoi: str) -> dict[str, str]:
    raw = ROOT / "data" / "raw"
    names = {
        "landsat": f"lst_landsat_{aoi}.tif",
        "sentinel": f"s2_lulc_{aoi}.tif",
        "era5": f"era5_{aoi}.tif",
        "ghsl": f"ghsl_{aoi}.tif",
        "terrain": f"terrain_{aoi}.tif",
    }
    return {k: str(raw / v) for k, v in names.items() if (raw / v).exists()}


def main(aoi: str = "new_delhi") -> None:
    paths = cached_paths(aoi)
    print(f"AOI {aoi}: aligning {list(paths)}")
    bands = align_stack(paths, ref="landsat")
    stack = derive_drivers(bands)
    df = drivers.stack_to_frame(stack)
    if "POP" not in df:
        df["POP"] = 1.0

    res = train_xgb(df, physics=True, cv=False, spatial_cv=True)
    m = res.metrics
    print(f"\nfeatures ({len(res.feature_names)}): {res.feature_names}")
    print(f"rows: {len(df)}")
    print(f"\nheld-out random split  R2={m['r2']:.4f}  "
          f"MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}")
    if res.metrics.get("r2_spatial") is not None:
        print(f"honest spatial-block   R2={m['r2_spatial']:.4f}  "
              f"(neighbours can't leak across train/test)")
    print("\ntop drivers:")
    for k, v in list(res.importance.items())[:8]:
        print(f"  {k:14s} {v:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "new_delhi")
