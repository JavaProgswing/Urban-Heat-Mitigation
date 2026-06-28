"""Smoke tests: pipeline pieces run + obey physics. Run: pytest -q"""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import drivers
from src.models.train import train_xgb
from src.scenarios import cooling


def _df():
    """Small deterministic frame with a known physical temperature signal."""
    n = 40
    rng = np.random.default_rng(7)
    row, col = np.indices((n, n))
    x, y = col / (n - 1), row / (n - 1)
    ndvi = np.clip(0.15 + 0.55 * y + rng.normal(0, 0.04, (n, n)), -1, 1)
    ndbi = np.clip(0.65 * x - 0.25 * ndvi + rng.normal(0, 0.04, (n, n)), -1, 1)
    built = np.clip(0.15 + 0.75 * x - 0.25 * y, 0, 1)
    albedo = np.clip(0.32 - 0.12 * built + 0.08 * ndvi, 0.05, 0.7)
    ndwi = np.clip(0.2 * ndvi - 0.25 * built, -1, 1)
    lst = (34 + 5.0 * ndbi + 4.0 * built - 3.5 * ndvi - 3.0 * albedo +
           rng.normal(0, 0.18, (n, n)))
    stack = {
        "NDVI": ndvi, "NDBI": ndbi, "NDWI": ndwi, "albedo": albedo,
        "build_frac": built, "BLD_H": 5 + 20 * built,
        "ELEV": 100 + 3 * y, "WATER_DIST": 1 + 20 * (1 - ndwi),
        "LST": lst,
    }
    df = drivers.stack_to_frame(stack)
    df["POP"] = 1.0 + 10.0 * df["build_frac"]
    return drivers.hotspots(df, pct=90.0)


def test_fixture_has_spatial_grid():
    df = _df()
    assert len(df) == 40 * 40
    assert {"row", "col", "LST", "hotspot"}.issubset(df.columns)


def test_xgb_learns_signal():
    res = train_xgb(_df(), cv=False)
    assert res.metrics["r2"] > 0.6           # recovers the planted signal
    assert "build_frac" in res.importance


def test_interventions_cool_not_warm():
    """Physics-constrained model: every cooling strategy must not warm."""
    df = _df()
    res = train_xgb(df, physics=True, cv=False)
    scen = cooling.simulate_all(res.model, res.feature_names, df,
                                {"cool_roofs": {"albedo_delta": 0.4,
                                                "applies_to": "built"},
                                 "urban_greening": {"ndvi_delta": 0.4,
                                                    "applies_to": "open"}})
    for name, r in scen.items():
        assert r.mean_cooling >= -1e-6, f"{name} predicted warming"


def test_physics_signs_complete():
    for c in drivers.DRIVER_COLS:
        assert c in drivers.PHYSICS_SIGNS


if __name__ == "__main__":
    test_fixture_has_spatial_grid()
    test_xgb_learns_signal()
    test_interventions_cool_not_warm()
    test_physics_signs_complete()
    print("all smoke tests passed")
