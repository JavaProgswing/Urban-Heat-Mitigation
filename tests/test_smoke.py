"""Smoke tests: pipeline pieces run + obey physics. Run: pytest -q"""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import drivers, synthetic
from src.models.train import train_xgb
from src.scenarios import cooling


def _df():
    stack = synthetic.make_grid(n=48)
    df = drivers.stack_to_frame(stack)
    df["POP"] = 1.0
    return drivers.hotspots(df, pct=90.0)


def test_synthetic_shapes():
    stack = synthetic.make_grid(n=32)
    assert stack["LST"].shape == (32, 32)
    assert set(drivers.DRIVER_COLS).issubset(stack.keys())


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
    test_synthetic_shapes()
    test_xgb_learns_signal()
    test_interventions_cool_not_warm()
    test_physics_signs_complete()
    print("all smoke tests passed")
