"""Plain-English findings + a cool-material reference table for the dashboard.

generate_insights() turns the computed Analysis into the kind of bullet summary a
decision-maker reads first ("driven by low-albedo roofs", "priority zones: …",
"cool roofs cut up to N °C"). MATERIALS is static literature (cost / durability /
indoor cooling are not in our satellite model) used for the comparison panel.
"""
from __future__ import annotations

import pandas as pd

# Friendly factor + strategy names (kept here so insights read naturally).
_DRIVER = {"Built-up": "low-albedo roofs & pavements", "Vegetation": "sparse vegetation",
           "Water": "lack of water", "Albedo (reflectivity)": "dark, low-albedo surfaces",
           "Elevation": "terrain", "Bare ground": "exposed bare ground"}
_STRAT = {"cool_roofs": "Cool roofs", "cool_pavements": "Cool pavements",
          "high_albedo_paint": "High-albedo paint", "green_roofs": "Green roofs",
          "urban_greening": "Urban greening", "water_body": "New water features"}


def generate_insights(a, zones: pd.DataFrame | None = None,
                      group_importance=None) -> list[str]:
    """Return a short list of headline findings from the analysis."""
    out: list[str] = []

    # 1. dominant driver(s)
    if group_importance is not None:
        grp = group_importance(a.res.importance)
        top = [k for k in grp][:2]
        phrase = " & ".join(_DRIVER.get(t, t.lower()) for t in top)
        out.append(f"High surface temperatures are driven mainly by {phrase}.")

    # 2. priority neighbourhoods
    if zones is not None and len(zones):
        pri = zones[zones["priority"]] if "priority" in zones else zones.head(3)
        names = list(pri["zone"].head(3))
        if names:
            out.append(f"{', '.join(names)} "
                       f"{'are' if len(names) > 1 else 'is'} the priority "
                       "heat-stress zone" + ("s." if len(names) > 1 else "."))

    # 3. best cooling strategy + magnitude
    if a.scenarios:
        best = max(a.scenarios.items(), key=lambda kv: kv[1].mean_cooling)
        bname, br = best
        peak = float(br.delta_lst[br.eligible].max()) if br.eligible.any() else br.mean_cooling
        out.append(f"{_STRAT.get(bname, bname)} can cut surface temperature by "
                   f"{br.mean_cooling:.2f}–{peak:.2f} °C where applied.")

    # 4. concrete recommendation
    if a.scenarios and zones is not None and len(zones):
        best = max(a.scenarios.items(), key=lambda kv: kv[1].mean_cooling)[0]
        hottest = zones.iloc[0]["zone"]
        share = float(a.scenarios[max(a.scenarios, key=lambda k: a.scenarios[k].mean_cooling)]
                      .eligible.mean())
        out.append(f"Recommended: deploy {_STRAT.get(best, best).lower()} across "
                   f"~{share:.2%} of {hottest} (the hottest zone) for maximum impact.")

    # 5. cooling potential spread
    if a.scenarios:
        vals = sorted((r.mean_cooling for r in a.scenarios.values()), reverse=True)
        if len(vals) >= 2:
            out.append(f"Across all strategies, mean achievable cooling ranges "
                       f"{vals[-1]:.2f}–{vals[0]:.2f} °C.")

    # 6. best value (cooling per cost) — the budget-minded pick
    if a.scenarios:
        ce = cost_effectiveness(a.scenarios)
        if len(ce):
            bv = ce.iloc[0]
            out.append(f"Best value for money: {bv['strategy'].lower()} — most "
                       f"cooling per rupee ({bv['cost'].lower()} cost, "
                       f"{bv['cooling_C']:.2f} °C).")
    return out


# Relative install cost per strategy (qualitative tiers -> weight) for the
# cost-effectiveness ranking. Decision-makers act on cooling-per-rupee, not raw
# cooling — cheap cool roofs usually beat costly green roofs / water bodies.
_COST_TIER = {"cool_roofs": ("Low", 1.0), "high_albedo_paint": ("Low", 1.0),
              "cool_pavements": ("Medium", 2.0), "urban_greening": ("Medium", 2.0),
              "green_roofs": ("High", 3.0), "water_body": ("High", 3.0)}


def cost_effectiveness(scenarios) -> pd.DataFrame:
    """Rank strategies by cooling delivered per unit relative cost (value).

    value = mean_cooling / cost_weight, normalised so the best = 100. Answers the
    planner's real question — where does each rupee buy the most cooling.
    """
    rows = []
    for k, r in scenarios.items():
        tier, w = _COST_TIER.get(k, ("Medium", 2.0))
        rows.append({"strategy": _STRAT.get(k, k), "cooling_C": round(r.mean_cooling, 2),
                     "cost": tier, "value": r.mean_cooling / w})
    df = pd.DataFrame(rows).sort_values("value", ascending=False)
    top = df["value"].max() or 1.0
    df["value_index"] = (100 * df["value"] / top).round(0).astype(int)
    return df.drop(columns="value").reset_index(drop=True)


# --- static cool-material reference (literature; not model output) ------------
MATERIALS = pd.DataFrame([
    {"Material": "Cool roof coating", "Albedo": 0.65, "Surface cooling (°C)": "8–14",
     "Indoor cooling (°C)": "2–4", "Rel. cost": "Low", "Durability (yr)": "10–15"},
    {"Material": "High-albedo paint", "Albedo": 0.55, "Surface cooling (°C)": "5–10",
     "Indoor cooling (°C)": "2–3", "Rel. cost": "Low", "Durability (yr)": "5–10"},
    {"Material": "Reflective pavement", "Albedo": 0.35, "Surface cooling (°C)": "3–8",
     "Indoor cooling (°C)": "—", "Rel. cost": "Medium", "Durability (yr)": "10–20"},
    {"Material": "Green roof", "Albedo": 0.30, "Surface cooling (°C)": "2–5",
     "Indoor cooling (°C)": "2–5", "Rel. cost": "High", "Durability (yr)": "20–40"},
    {"Material": "Tree canopy / greening", "Albedo": 0.20, "Surface cooling (°C)": "2–8",
     "Indoor cooling (°C)": "1–3", "Rel. cost": "Medium", "Durability (yr)": "30+"},
    {"Material": "Water body / wetland", "Albedo": 0.08, "Surface cooling (°C)": "3–5",
     "Indoor cooling (°C)": "—", "Rel. cost": "High", "Durability (yr)": "—"},
])
