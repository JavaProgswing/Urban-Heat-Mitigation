"""Simulate cooling interventions + optimize placement.

Method (counterfactual): take observed driver frame -> apply an intervention's
physical edit to eligible pixels (e.g. cool roof => +albedo on built pixels) ->
re-predict LST with the trained (physics-informed) model -> deltaLST = before - after.
Because the model is physics-constrained, these extrapolated states stay sane.

Optimizer: greedy pick of the strategy giving max cooling per eligible pixel,
subject to a coverage budget, weighted by population exposure (heat-risk).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Physical "achievable" targets for albedo / vegetation interventions. Cooling
# scales with how far a pixel is BELOW the target (a dark roof can be whitened a
# lot; an already-bright one can't), so the per-pixel estimate is realistic.
ALBEDO_TARGET = 0.60      # typical cool-roof / high-albedo finish
NDVI_TARGET = 0.70        # dense healthy canopy


# Eligibility: which pixels a strategy can apply to. Thresholds are ADAPTIVE
# (per-AOI quantiles) not fixed constants, so a strategy is never empty just
# because a city's absolute NDVI/built distribution differs.
#
# CRITICAL: "built" and "open" are made LAND-USE EXCLUSIVE by build_frac (split
# at the median). A roof intervention only makes sense where there are roofs;
# tree planting / water only where there is open ground. Without the build_frac
# gate on "open", greening was eligible on dense rooftops, and since cool roofs
# deliver a flat (physics-anchored) cooling that beats variable greening, the
# optimizer chose cool roofs almost everywhere. The gate restores a realistic
# split: roofs in the built core, greening/water on open land.
def _eligible(df: pd.DataFrame, applies_to: str) -> np.ndarray:
    bf = df["build_frac"].to_numpy()
    ndvi = df["NDVI"].to_numpy()
    bf_mid = np.quantile(bf, 0.50)
    if applies_to == "built":                 # roofs: the more-built half
        return bf >= bf_mid
    if applies_to == "impervious":            # pavements: built but unvegetated
        return (df["NDBI"].to_numpy() >= np.quantile(df["NDBI"], 0.50)) & \
               (ndvi <= np.quantile(ndvi, 0.50))
    if applies_to == "open":                  # greening: plantable open land
        return (bf < bf_mid) & (ndvi <= np.quantile(ndvi, 0.60))
    if applies_to == "water":                 # new/expanded water: feasible only
        # on open land NEAR existing water (you can't put a lake anywhere).
        wd = df["WATER_DIST"].to_numpy() if "WATER_DIST" in df else np.zeros(len(df))
        return (bf < bf_mid) & (wd <= np.quantile(wd, 0.30))
    return np.ones(len(df), dtype=bool)


def _refresh_context(out: pd.DataFrame, edited: list[str], shape) -> None:
    """Recompute neighbourhood-context (_N) features for edited drivers in place.

    A broad intervention (e.g. greening the low-veg ~60% of the grid) raises not
    just a pixel's own NDVI but its neighbourhood mean NDVI_N — a top model
    driver. Without this refresh the model sees local NDVI rise but NDVI_N flat,
    so it under-predicts cooling. Regrid edited driver -> area mean -> back to
    per-pixel, matching how align.spatial_drivers builds the feature originally.
    """
    if not edited or shape is None:
        return
    try:
        from ..data.align import _nanmean_filter, CONTEXT_SIZE
    except Exception:
        return
    rows, cols = out["row"].to_numpy(), out["col"].to_numpy()
    for c in edited:
        nc = f"{c}_N"
        if nc not in out:
            continue
        grid = np.full(shape, np.nan, "float32")
        grid[rows, cols] = out[c].to_numpy("float32")
        out[nc] = _nanmean_filter(grid, CONTEXT_SIZE)[rows, cols]


def apply_intervention(df: pd.DataFrame, spec: dict, shape=None) -> pd.DataFrame:
    """Return a copy of df with driver columns edited per the strategy spec.
    When `shape` is given, neighbourhood-context (_N) features are recomputed so
    the intervention's area effect reaches the model (see _refresh_context)."""
    out = df.copy()
    mask = _eligible(out, spec.get("applies_to", "all"))
    edited: list[str] = []
    if "albedo_delta" in spec and "albedo" in out:
        out.loc[mask, "albedo"] = np.clip(
            out.loc[mask, "albedo"] + spec["albedo_delta"], 0.05, 0.9)
        edited.append("albedo")
    if "ndvi_delta" in spec and "NDVI" in out:
        out.loc[mask, "NDVI"] = np.clip(
            out.loc[mask, "NDVI"] + spec["ndvi_delta"], -1, 1)
        edited.append("NDVI")
        if "build_frac" in out:                    # greening reduces built frac
            out.loc[mask, "build_frac"] = np.clip(
                out.loc[mask, "build_frac"] - 0.2 * spec["ndvi_delta"], 0, 1)
            edited.append("build_frac")
    if "ndwi_delta" in spec and "NDWI" in out:
        out.loc[mask, "NDWI"] = np.clip(
            out.loc[mask, "NDWI"] + spec["ndwi_delta"], -1, 1)
        edited.append("NDWI")
    _refresh_context(out, edited, shape)
    return out, mask


@dataclass
class ScenarioResult:
    name: str
    delta_lst: np.ndarray          # per-pixel cooling (deg C), 0 where ineligible
    eligible: np.ndarray
    mean_cooling: float            # mean over eligible pixels
    pixels: int


def humidity_evap_factor(rh, rh_ref: float = 45.0, lo: float = 0.5,
                         hi: float = 1.25) -> float:
    """Climate scaling for EVAPORATIVE cooling (greening, water bodies).

    Evapotranspiration / evaporative cooling scales with the vapour-pressure
    deficit, which falls as relative humidity rises — so trees and ponds cool a
    humid coastal city (Chennai ~70% RH) far less than a dry one (Delhi ~40%).
    Albedo strategies reflect sunlight and are humidity-independent, so they are
    NOT scaled. rh_ref ~45% (semi-arid) is the 1.0 reference.
    """
    if rh is None:
        return 1.0
    ratio = (1.0 - float(rh) / 100.0) / (1.0 - rh_ref / 100.0)
    return float(np.clip(ratio, lo, hi))


def simulate(model, feature_names, df: pd.DataFrame, name: str,
             spec: dict, lst_per_albedo: float = 12.0, shape=None,
             base=None, lst_per_ndwi: float = 6.0, lst_per_ndvi: float = 8.0,
             max_cooling_C: float = 8.0, evap_factor: float = 1.0) -> ScenarioResult:
    """ΔLST from an intervention: the ML counterfactual response, floored at a
    literature prior for drivers the correlational model handles unreliably.

      - greening (NDVI): ML response is reliable -> use it directly.
      - albedo (cool roofs/pavements/paint): confounded with NDBI/NDVI, so floor
        at lst_per_albedo * albedo_delta.
      - water (NDWI): positively tied to LST in dry urban scenes -> the physics
        monotone constraint fights the data and XGBoost enforcement leaks, so
        floor at lst_per_ndwi * ndwi_delta.

    Floors use max(), NOT addition: adding the ML delta to the prior double-counts
    when the model *does* respond (cool_roofs otherwise stacks to ~12 deg C).

    `base` (unintervened prediction) is identical across strategies — pass it in
    from simulate_all to avoid recomputing it once per scenario.
    """
    if base is None:
        base = model.predict(df[feature_names].to_numpy("float32"))
    edited, mask = apply_intervention(df, spec, shape=shape)
    after = model.predict(edited[feature_names].to_numpy("float32"))
    # Clamp to >=0 (interventions can't warm; guards the NDWI monotone leak).
    ml = np.maximum(base - after, 0.0)
    # Physics floors use the ACTUAL achievable change per pixel, not a flat delta:
    # whitening a dark roof gains more than an already-light one; greening bare
    # ground gains more than semi-vegetated land. So cooling = prior x (target -
    # current), clipped to the strategy's nominal delta. This makes the estimate
    # spatially varying and physical, not a flat per-strategy constant.
    albedo_d = float(spec.get("albedo_delta", 0.0))
    if albedo_d > 0 and "albedo" in df:
        gain = np.clip(ALBEDO_TARGET - df["albedo"].to_numpy(), 0.0, albedo_d)
        ml = np.maximum(ml, lst_per_albedo * gain)
    ndvi_d = float(spec.get("ndvi_delta", 0.0))
    if ndvi_d > 0 and "NDVI" in df:                             # greening / green roofs
        gain = np.clip(NDVI_TARGET - df["NDVI"].to_numpy(), 0.0, ndvi_d)
        ml = np.maximum(ml, lst_per_ndvi * evap_factor * gain)  # evaporative
    ndwi_d = float(spec.get("ndwi_delta", 0.0))
    if ndwi_d > 0:                                               # water: evaporative
        ml = np.maximum(ml, lst_per_ndwi * evap_factor * ndwi_d)
    # Realism cap: model extrapolates implausibly on extreme hot/bare pixels
    # (e.g. 17 deg C from one greening step). One intervention realistically
    # delivers <= max_cooling_C; clip so per-pixel and mean values stay credible.
    delta = np.where(mask, np.minimum(ml, max_cooling_C), 0.0)   # +ve = cooling
    elig = int(mask.sum())
    return ScenarioResult(
        name=name, delta_lst=delta, eligible=mask,
        mean_cooling=float(delta[mask].mean()) if elig else 0.0, pixels=elig)


def simulate_all(model, feature_names, df, scenarios: dict[str, dict],
                 lst_per_albedo: float = 12.0, shape=None, lst_per_ndwi: float = 6.0,
                 lst_per_ndvi: float = 8.0, max_cooling_C: float = 8.0,
                 evap_factor: float = 1.0):
    base = model.predict(df[feature_names].to_numpy("float32"))  # once, shared
    return {name: simulate(model, feature_names, df, name, spec,
                           lst_per_albedo, shape=shape, base=base,
                           lst_per_ndwi=lst_per_ndwi, lst_per_ndvi=lst_per_ndvi,
                           max_cooling_C=max_cooling_C, evap_factor=evap_factor)
            for name, spec in scenarios.items()}


def optimize(results: dict[str, ScenarioResult], df: pd.DataFrame,
             budget_frac: float = 0.3, pop_col: str = "POP") -> pd.DataFrame:
    """Per-pixel best strategy = the full intervention recommendation map.

    For every pixel, pick the strategy that cools it most (strategies are
    land-use-exclusive, so this is genuinely diverse: roofs in the built core,
    greening/water on open land). Returns ALL pixels with any cooling, ranked by
    population-weighted cooling, with a `priority` flag marking the top
    `budget_frac` (where to act FIRST). The map shows the whole recommendation;
    the priority subset drives the "act first" table + headline metrics.
    """
    n = len(df)
    best_name = np.full(n, "none", dtype=object)
    best_cool = np.zeros(n)
    for name, r in results.items():
        better = r.delta_lst > best_cool
        best_cool = np.where(better, r.delta_lst, best_cool)
        best_name = np.where(better, name, best_name)

    pop = df[pop_col].to_numpy() if pop_col in df else np.ones(n)
    score = best_cool * (1.0 + np.log1p(np.clip(pop, 0, None)))
    has = best_cool > 0                                  # pixel has a useful action
    order = np.argsort(-score)
    order = order[has[order]]                            # all actionable, ranked
    k = max(1, int(budget_frac * has.sum()))

    out = pd.DataFrame({
        "row": df["row"].to_numpy()[order] if "row" in df else order,
        "col": df["col"].to_numpy()[order] if "col" in df else 0,
        "best_strategy": best_name[order],
        "cooling_C": best_cool[order],
        "pop": pop[order],
        "score": score[order],
    }).reset_index(drop=True)
    out["priority"] = out.index < k                      # top budget_frac = act first
    return out
