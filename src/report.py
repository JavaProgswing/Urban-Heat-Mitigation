"""One-page analysis report (PNG + PDF) for submissions.

Renders a single landscape sheet: header + metrics, LST hotspot map, driver
importance, cooling potential, and the optimized intervention placement. Used by
the CLI (writes outputs/report.{png,pdf}) and the dashboard (one-click download).
"""
from __future__ import annotations
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless / thread-safe rendering
import matplotlib.pyplot as plt             # noqa: E402
import numpy as np                          # noqa: E402

from .features import drivers               # noqa: E402
from .viz import maps                       # noqa: E402

SCEN_COLORS = {
    "cool_roofs": "#4FC3F7", "cool_pavements": "#BA68C8",
    "high_albedo_paint": "#4DD0E1", "green_roofs": "#AED581",
    "urban_greening": "#66BB6A", "water_body": "#1E88E5",
}
SCEN_LABEL = {
    "cool_roofs": "Cool roofs", "cool_pavements": "Cool pavements",
    "high_albedo_paint": "High-albedo paint", "green_roofs": "Green roofs",
    "urban_greening": "Urban greening", "water_body": "New water features",
}
DRIVER_LABEL = {
    "NDVI": "Vegetation", "NDBI": "Built-up", "NDWI": "Water", "albedo": "Albedo",
    "build_frac": "Building density", "BLD_H": "Building height",
    "ELEV": "Elevation", "WATER_DIST": "Dist. to water",
    "NDVI_STD": "Veg. texture", "AIR_T": "Air temp", "RH": "Humidity",
    "WIND": "Wind",
}


def _dlabel(k: str) -> str:
    if k.startswith("LULC_"):
        return f"{k[5:].capitalize()} cover"
    if k.endswith("_STD"):
        return f"{DRIVER_LABEL.get(k[:-4], k[:-4])} texture"
    if k.endswith("_NC"):
        return f"{DRIVER_LABEL.get(k[:-3], k[:-3])} (district)"
    if k.endswith("_N"):
        return f"{DRIVER_LABEL.get(k[:-2], k[:-2])} (area)"
    return DRIVER_LABEL.get(k, k)


def _group_importance(importance: dict) -> dict:
    """Sum SHAP importance by physical factor for a clean report chart."""
    def grp(f):
        if f.startswith("LULC_"):
            return {"tree": "Vegetation", "grass": "Vegetation",
                    "crop": "Vegetation", "shrub": "Vegetation",
                    "built": "Built-up", "water": "Water",
                    "bare": "Bare ground"}.get(f[5:], "Land cover")
        b = (f[:-4] if f.endswith("_STD") else f[:-3] if f.endswith("_NC")
             else f[:-2] if f.endswith("_N") else f)
        return {"NDVI": "Vegetation", "NDBI": "Built-up",
                "build_frac": "Built-up", "BLD_H": "Built-up", "NDWI": "Water",
                "WATER_DIST": "Water", "albedo": "Albedo", "ELEV": "Elevation"}.get(b, b)
    agg = {}
    for k, v in importance.items():
        agg[grp(k)] = agg.get(grp(k), 0.0) + float(v)
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))


def _placement_rgb(plan, shape):
    import matplotlib.colors as mcolors
    code = {s: i for i, s in enumerate(SCEN_COLORS)}
    g = maps.regrid(plan, plan["best_strategy"].map(code).to_numpy("float32"), shape)
    g = np.nan_to_num(g, nan=-1.0)
    img = np.full((*shape, 3), 0.06)
    for s, i in code.items():
        img[g == i] = mcolors.to_rgb(SCEN_COLORS[s])
    return img


def build_figure(analysis, cfg):
    """Compose the report as a single matplotlib Figure."""
    from matplotlib.gridspec import GridSpec
    res, df, shape = analysis.res, analysis.df, analysis.shape
    m = res.metrics
    sp = m.get("r2_spatial")

    fig = plt.figure(figsize=(11.0, 8.5), facecolor="white")
    gs = GridSpec(3, 2, height_ratios=[0.62, 2.0, 2.0],
                  hspace=0.42, wspace=0.22, figure=fig)

    # ---- header ----
    axh = fig.add_subplot(gs[0, :]); axh.axis("off")
    bbox = [round(b, 3) for b in cfg.bbox]
    axh.text(0, 0.78, "Urban Heat Mitigation — AI/ML Decision Support",
             fontsize=17, fontweight="bold")
    atm = getattr(analysis, "atmosphere", {}) or {}
    atm_s = (f"   ·   ERA5: {atm.get('air_temp_C', '—')}°C air, "
             f"{atm.get('humidity_pct', '—')}% RH, {atm.get('wind_m_s', '—')} m/s"
             if atm else "")
    axh.text(0, 0.40,
             f"AOI: {cfg.aoi_name}   bbox={bbox}   dates {cfg.start} → {cfg.end}"
             f"   ·   source: live satellite   model: XGBoost{atm_s}",
             fontsize=9.0, color="#444")
    honest = f"{sp:.3f}" if sp is not None else "—"
    top = SCEN_LABEL.get(analysis.plan["best_strategy"].mode().iat[0], "—")
    axh.text(0, 0.02,
             f"Honest R² (unseen ground): {honest}     In-scene R²: {m['r2']:.3f}"
             f"     MAE: {m['mae']:.2f} °C     RMSE: {m['rmse']:.2f} °C"
             f"     ·   Top strategy: {top}   "
             f"(mean planned cooling {analysis.plan['cooling_C'].mean():.2f} °C "
             f"over {len(analysis.plan):,} cells)",
             fontsize=9.5, color="#222")

    # ---- LST hotspot map ----
    ax1 = fig.add_subplot(gs[1, 0])
    lst = maps.regrid(df, df[drivers.TARGET_COL].to_numpy(), shape)
    im = ax1.imshow(lst, cmap="inferno")
    if "hotspot" in df:
        from scipy import ndimage
        hot = np.nan_to_num(maps.regrid(df, df["hotspot"].to_numpy("float32"),
                                        shape)) > 0.5
        ax1.contour(hot, levels=[0.5], colors="#00e5ff", linewidths=0.6)
    ax1.set_title("Land surface temperature + hotspots", fontsize=11)
    ax1.axis("off")
    fig.colorbar(im, ax=ax1, shrink=0.8, label="°C")

    # ---- driver importance (grouped by physical factor) ----
    ax2 = fig.add_subplot(gs[1, 1])
    grp = _group_importance(res.importance)
    items = list(grp.items())[::-1]
    ax2.barh([k for k, _ in items], [v for _, v in items], color="#6db3f2")
    ax2.set_title("Heat drivers (SHAP, by factor)", fontsize=11)
    ax2.tick_params(labelsize=9)

    # ---- cooling potential ----
    ax3 = fig.add_subplot(gs[2, 0])
    sc = sorted(analysis.scenarios.items(), key=lambda kv: kv[1].mean_cooling)
    ax3.barh([SCEN_LABEL.get(k, k) for k, _ in sc],
             [v.mean_cooling for _, v in sc],
             color=[SCEN_COLORS.get(k, "#5ec8a8") for k, _ in sc])
    ax3.set_title("Cooling potential by strategy (°C)", fontsize=11)
    ax3.tick_params(labelsize=8)

    # ---- placement map ----
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.imshow(_placement_rgb(analysis.plan, shape))
    ax4.set_title("Where to act — recommended strategy per priority cell",
                  fontsize=11)
    ax4.axis("off")
    present = [s for s in SCEN_COLORS if s in set(analysis.plan["best_strategy"])]
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=7,
                          markerfacecolor=SCEN_COLORS[s], markeredgecolor="none",
                          label=SCEN_LABEL[s]) for s in present]
    if handles:
        ax4.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, -0.18))

    lv = getattr(analysis, "lst_validation", None)
    val_s = (f"  ·  ECOSTRESS cross-sensor: r={lv['pearson_r']}, "
             f"MAE={lv['mae_C']}°C" if lv else "")
    fig.text(0.5, 0.005, "Generated by Urban Heat AI · physics-informed ML on "
             "Landsat 8+9 + Sentinel-2 + ERA5 + GHSL + SRTM" + val_s, ha="center",
             fontsize=7.5, color="#888")
    return fig


def save_report(analysis, cfg, out_dir, stem: str = "report"):
    out_dir = Path(out_dir)
    fig = build_figure(analysis, cfg)
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def figure_bytes(analysis, cfg, fmt: str = "png") -> bytes:
    """Render the report to in-memory bytes (for a dashboard download button)."""
    fig = build_figure(analysis, cfg)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
