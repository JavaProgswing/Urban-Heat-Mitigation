"""Urban Heat AI — interactive dashboard.

Pick any city/region by name (or draw a box on the map), pick a date window,
choose Demo (synthetic) or Live (Earth Engine satellite), and run the full
analysis: heat hotspots, drivers, cooling scenarios, optimized plan.

    streamlit run dashboard/app.py
"""
from __future__ import annotations
import datetime as dt
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features import drivers
from src.pipeline import run_analysis
from src.viz import maps

st.set_page_config(page_title="Urban Heat AI", layout="wide")

BASE = load_config()

st.markdown("""
<style>
  footer, #MainMenu {visibility:hidden;}
  .block-container {padding-top:2rem; padding-bottom:2.5rem; max-width:1380px;}
  h1,h2,h3,h4 {letter-spacing:-0.015em; font-weight:650;}
  [data-testid="stMetricValue"] {font-size:1.55rem; font-weight:650;}
  [data-testid="stMetricLabel"] p {color:#8b97a7; font-size:.82rem;}
  [data-testid="stCaptionContainer"], .stCaption {color:#7b8794; font-size:.82rem; line-height:1.45;}
  [data-testid="stSidebar"] {border-right:1px solid rgba(255,255,255,.06);}
  [data-testid="stSidebar"] .block-container {padding-top:1.2rem;}
  .stTabs [data-baseweb="tab-list"] {gap:4px;}
  .stTabs [data-baseweb="tab"] {padding:6px 14px;}
  hr {margin:.7rem 0; border-color:rgba(255,255,255,.07);}
</style>
""", unsafe_allow_html=True)

# --------------------------- presentation helpers -------------------------- #
ACCENT = "#6db3f2"
_PRETTY = {
    "NDVI": "Vegetation", "NDBI": "Built-up", "NDWI": "Water", "albedo": "Albedo",
    "build_frac": "Building density", "ELEV": "Elevation",
    "WATER_DIST": "Dist. to water", "NDVI_STD": "Veg. texture",
    "AIR_T": "Air temp", "RH": "Humidity", "WIND": "Wind",
}
_SCEN = {
    "cool_roofs": "Cool roofs", "cool_pavements": "Cool pavements",
    "high_albedo_paint": "High-albedo paint", "green_roofs": "Green roofs",
    "urban_greening": "Urban greening", "water_body": "Water bodies",
}

# Group the 26 raw features by the physical factor they represent, so the
# "what drives heat" chart reads as ~6 meaningful levers, not 26 cryptic bars.
def _group_of(feat: str) -> str:
    if feat.startswith("LULC_"):
        return {"tree": "Vegetation", "grass": "Vegetation", "crop": "Vegetation",
                "shrub": "Vegetation", "built": "Built-up", "water": "Water",
                "bare": "Bare ground"}.get(feat[5:], "Land cover")
    base = (feat[:-4] if feat.endswith("_STD") else feat[:-3] if feat.endswith("_NC")
            else feat[:-2] if feat.endswith("_N") else feat)
    return {"NDVI": "Vegetation",
            "NDBI": "Built-up", "build_frac": "Built-up", "BLD_H": "Built-up",
            "NDWI": "Water", "WATER_DIST": "Water",
            "albedo": "Albedo (reflectivity)", "ELEV": "Elevation"}.get(base, base)


def group_importance(importance: dict) -> dict:
    agg: dict[str, float] = {}
    for k, v in importance.items():
        agg[_group_of(k)] = agg.get(_group_of(k), 0.0) + float(v)
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))


def pretty(name: str) -> str:
    """Human label for a raw driver / context / land-cover feature."""
    if name.startswith("LULC_"):
        return f"{name[5:].capitalize()} cover"
    if name.endswith("_STD"):
        return f"{_PRETTY.get(name[:-4], name[:-4])} texture"
    if name.endswith("_NC"):
        return f"{_PRETTY.get(name[:-3], name[:-3])} (district)"
    if name.endswith("_N"):
        return f"{_PRETTY.get(name[:-2], name[:-2])} (area)"
    return _PRETTY.get(name, name)


def hbar(data, value, label, fmt=".2f", color=ACCENT):
    """Horizontal bar chart — readable horizontal labels, transparent, themed."""
    h = max(150, 30 * len(data))
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=3, color=color, opacity=0.92)
        .encode(
            x=alt.X(f"{value}:Q", title=None),
            y=alt.Y(f"{label}:N", sort="-x", title=None),
            tooltip=[alt.Tooltip(f"{label}:N", title=""),
                     alt.Tooltip(f"{value}:Q", format=fmt, title="")],
        )
        .properties(height=h)
        .configure_axis(labelColor="#cbd5e1", titleColor="#cbd5e1",
                        labelFontSize=13, gridColor="rgba(255,255,255,0.06)",
                        domainColor="rgba(255,255,255,0.15)", tickColor="rgba(255,255,255,0.15)")
        .configure_view(strokeWidth=0)
    )


def colorbar_html(lo, hi):
    return (
        '<div style="margin-top:6px">'
        '<div style="height:10px;border-radius:6px;background:linear-gradient('
        '90deg,#000004,#3b0f70,#8c2981,#de4968,#fe9f6d,#fcfdbf)"></div>'
        '<div style="display:flex;justify-content:space-between;color:#94a3b8;'
        f'font-size:12px;margin-top:4px"><span>{lo:.1f} °C · cooler</span>'
        f'<span>hotter · {hi:.1f} °C</span></div></div>'
    )


# distinct colour per intervention strategy (placement map + legend)
SCEN_COLORS = {
    "cool_roofs": "#4FC3F7", "cool_pavements": "#BA68C8",
    "high_albedo_paint": "#4DD0E1", "green_roofs": "#AED581",
    "urban_greening": "#66BB6A", "water_body": "#1E88E5",
}


def placement_image(plan, shape):
    """RGB map of the optimized plan: each priority cell coloured by its chosen
    strategy, the rest dark. Shows the *spatial placement* of interventions."""
    import matplotlib.colors as mcolors
    code = {s: i for i, s in enumerate(SCEN_COLORS)}
    g = maps.regrid(plan, plan["best_strategy"].map(code).to_numpy("float32"), shape)
    g = np.nan_to_num(g, nan=-1.0)
    img = np.full((*shape, 3), 14, "uint8")            # near-black background
    for s, i in code.items():
        img[g == i] = [int(255 * c) for c in mcolors.to_rgb(SCEN_COLORS[s])]
    return img


def legend_html(strategies):
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;'
        f'color:#cbd5e1;font-size:12px"><span style="width:11px;height:11px;'
        f'border-radius:3px;background:{SCEN_COLORS[s]};margin-right:5px;'
        f'display:inline-block"></span>{_SCEN.get(s, s)}</span>'
        for s in strategies)
    return f'<div style="margin-top:6px">{chips}</div>'


def validation_chart(eval_d):
    """Predicted vs observed LST scatter with 1:1 line — visual model validation."""
    yt = np.asarray(eval_d["y_true"], "float32")
    yp = np.asarray(eval_d["y_pred"], "float32")
    if len(yt) > 3000:
        idx = np.random.default_rng(0).choice(len(yt), 3000, replace=False)
        yt, yp = yt[idx], yp[idx]
    d = pd.DataFrame({"observed": yt, "predicted": yp})
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    pts = (alt.Chart(d).mark_circle(size=12, opacity=0.22, color=ACCENT)
           .encode(x=alt.X("observed:Q", title="Observed LST °C",
                           scale=alt.Scale(domain=[lo, hi])),
                   y=alt.Y("predicted:Q", title="Predicted LST °C",
                           scale=alt.Scale(domain=[lo, hi]))))
    line = (alt.Chart(pd.DataFrame({"x": [lo, hi], "y": [lo, hi]}))
            .mark_line(color="#e0588a", strokeDash=[5, 4])
            .encode(x="x:Q", y="y:Q"))
    return ((pts + line).properties(height=340)
            .configure_axis(labelColor="#cbd5e1", titleColor="#cbd5e1",
                            gridColor="rgba(255,255,255,0.06)",
                            domainColor="rgba(255,255,255,0.15)")
            .configure_view(strokeWidth=0))


def bbox_km(bb):
    """Approx width/height of a lon/lat bbox in km (for AOI-size feedback)."""
    midlat = (bb[1] + bb[3]) / 2.0
    w = abs(bb[2] - bb[0]) * 111.32 * float(np.cos(np.radians(midlat)))
    h = abs(bb[3] - bb[1]) * 110.57
    return w, h


st.markdown(
    "<div style='display:flex;align-items:baseline;gap:12px;margin:.1rem 0 1.1rem'>"
    "<span style='font-size:1.7rem;font-weight:700;letter-spacing:-.02em'>"
    "Urban Heat Mitigation</span>"
    "<span style='color:#7b8794;font-size:.95rem'>AI/ML decision support</span>"
    "</div>",
    unsafe_allow_html=True,
)

def _torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------- sidebar: inputs ----------------------------- #
# One seamless AOI picker: search a place OR draw a rectangle on the map. No
# coordinate typing, no mode switch. A drawn box overrides the searched one
# until you search again or reset.
st.sidebar.subheader("Region")


@st.cache_data(show_spinner=False)
def _geocode(place: str, size_km: int):
    from src.geocode import geocode_aoi
    return geocode_aoi(place, max_km=size_km)


place = st.sidebar.text_input(
    "Search a place", value="Delhi",
    help="City, district, locality or landmark — e.g. 'IIT Delhi', "
         "'Hauz Khas'. Press Enter.")
size_km = st.sidebar.slider("Area size (km)", 1, 60, 10,
                            help="Box around the place centre. 2–5 km suits a "
                                 "single locality.")

# Map key tracks place+size: the map remounts/re-centres on a deliberate search
# change, but stays stable while you draw so the rectangle is never lost.
map_key = f"aoi::{place.strip().lower()}::{size_km}"

# A new search (place or size) clears any drawn override.
if st.session_state.get("aoi_key") != (place, size_km):
    st.session_state.pop("draw_override", None)
    st.session_state["aoi_key"] = (place, size_km)

# Geocode the searched place (cached) -> base box.
geo_bbox, geo_name = None, None
if place.strip():
    try:
        geo_bbox, geo_name, _ = _geocode(place.strip(), size_km)
    except Exception as e:
        st.sidebar.error(f"Couldn't find “{place}”: {e}")

# Resolve the active AOI: a drawn box wins, else the searched box, else default.
override = st.session_state.get("draw_override")
if override:
    bbox, aoi_name = override, f"{geo_name or 'area'}_custom"
    w_km, h_km = bbox_km(bbox)
    st.sidebar.success(f"Drawn box · {w_km:.1f} × {h_km:.1f} km")
    if st.sidebar.button("Use searched place", use_container_width=True):
        st.session_state.pop("draw_override", None)
        st.rerun()
elif geo_bbox:
    bbox, aoi_name = geo_bbox, geo_name
    w_km, h_km = bbox_km(bbox)
    st.sidebar.success(f"{geo_name} · {w_km:.1f} × {h_km:.1f} km")
else:
    bbox, aoi_name = list(BASE.bbox), BASE.aoi_name

if min(bbox_km(bbox)) < 1.5:
    st.sidebar.warning("Small AOI (<1.5 km) — noisier R² with little spatial "
                       "variety. 2–5 km works best.")

st.sidebar.caption("Or draw a rectangle on the map to analyse your own area.")

st.sidebar.subheader("Dates")
today = dt.date.today()
end = st.sidebar.date_input("End", value=today)
start = st.sidebar.date_input("Start", value=today - dt.timedelta(days=120))
st.sidebar.caption("A 3–4 month window gives a cloud-free composite.")

st.sidebar.subheader("Data & model")
source = st.sidebar.radio("Data source", ["Demo (synthetic)", "Live satellite (GEE)"])
src = "synthetic" if source.startswith("Demo") else "gee"
project = st.sidebar.text_input("GEE project id", value=BASE.gee_project,
                                disabled=(src == "synthetic"))
model_choices = ["xgb"] + (["pinn"] if _torch_ok() else [])
model = st.sidebar.selectbox("Model", model_choices)

run = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

# --------------------------- region preview map ---------------------------- #
def preview_map(bb, draw=False):
    import folium
    from folium.plugins import Draw, LocateControl
    span = max(abs(bb[2] - bb[0]), abs(bb[3] - bb[1]))
    zoom = 14 if span < 0.05 else 12 if span < 0.2 else 11 if span < 0.5 else 10
    m = folium.Map(location=[(bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2],
                   zoom_start=zoom, tiles="CartoDB positron")
    folium.Rectangle([[bb[1], bb[0]], [bb[3], bb[2]]],
                     color="#e74c3c", fill=True, fill_opacity=0.1).add_to(m)
    # 📍 GPS button: centres the map on the user's current location (browser
    # geolocation) so they can draw a box over their own locality / college.
    LocateControl(strings={"title": "Show my location"},
                  flyTo=True, keepCurrentZoomLevel=False).add_to(m)
    if draw:
        Draw(draw_options={"rectangle": {"shapeOptions": {"color": "#6db3f2"}},
                           "polygon": False, "polyline": False, "circle": False,
                           "marker": False, "circlemarker": False},
             edit_options={"edit": False, "remove": False}).add_to(m)
    return m


_BASE_TILES = {
    "Streets": ("CartoDB positron", None),
    "Satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}", "Esri World Imagery"),
    "Dark": ("CartoDB dark_matter", None),
}


def _downsample(rgba, cap=700):
    """Cap the longest side so the base64-embedded overlay stays light."""
    h, w = rgba.shape[:2]
    step = max(1, int(max(h, w) / cap))
    return rgba[::step, ::step] if step > 1 else rgba


def overlay_map(bbox, rgba, opacity=0.7, base="Streets"):
    """Folium map: model-output RGBA overlaid on a base map, projected to the AOI
    bounds (geographically correct — no raster stretching)."""
    import folium
    from folium.raster_layers import ImageOverlay
    minx, miny, maxx, maxy = bbox
    span = max(abs(maxx - minx), abs(maxy - miny))
    zoom = 14 if span < 0.05 else 13 if span < 0.12 else 12 if span < 0.3 else 11
    tiles, attr = _BASE_TILES.get(base, _BASE_TILES["Streets"])
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2],
                   zoom_start=zoom, tiles=tiles, attr=attr)
    ImageOverlay(image=_downsample(np.asarray(rgba, "float32")),
                 bounds=[[miny, minx], [maxy, maxx]], opacity=float(opacity),
                 mercator_project=True).add_to(m)
    folium.Rectangle([[miny, minx], [maxy, maxx]], color="#e74c3c",
                     weight=1, fill=False).add_to(m)
    return m


def _drawn_bbox(out):
    """Pull a [minx,miny,maxx,maxy] from st_folium's last drawn rectangle."""
    feat = (out or {}).get("last_active_drawing")
    if not feat or feat.get("geometry", {}).get("type") != "Polygon":
        return None
    ring = feat["geometry"]["coordinates"][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return [min(xs), min(ys), max(xs), max(ys)]


left, right = st.columns([1, 1])
with left:
    st.subheader("Selected region")
    out = None
    try:
        from streamlit_folium import st_folium
        # Stable key + restricted return: without a key the component remounts
        # whenever bbox changes and the freshly-drawn rectangle is lost before we
        # can read it (the "draw resets to default city" bug). The fixed key keeps
        # the drawing across reruns; returned_objects limits reruns to real draws.
        out = st_folium(preview_map(bbox, draw=True), height=360,
                        use_container_width=True, key=map_key,
                        returned_objects=["last_active_drawing"])
    except Exception:
        st.map(pd.DataFrame({"lat": [(bbox[1] + bbox[3]) / 2],
                             "lon": [(bbox[0] + bbox[2]) / 2]}))
    # A drawn rectangle becomes the active AOI (overrides the searched place).
    # The stable map_key keeps the drawing across the rerun, so this applies once
    # and settles without a loop.
    nb = _drawn_bbox(out)
    if nb:
        cur = st.session_state.get("draw_override")
        if cur is None or any(abs(a - b) > 1e-5 for a, b in zip(nb, cur)):
            st.session_state["draw_override"] = nb
            st.rerun()
    st.caption(f"AOI **{aoi_name}** · bbox={[round(b, 3) for b in bbox]} · "
               f"dates {start} → {end}")


# ------------------------------ run analysis ------------------------------- #
@st.cache_data(show_spinner=False)
def analyze(bbox, name, start, end, src, model, project):
    cfg = BASE.override(bbox=bbox, name=name, start=str(start), end=str(end),
                        project=project)
    a = run_analysis(cfg, source=src, model=model, cv=(src == "synthetic"))
    # return plain data (cache-friendly) + keep model for live scenario tweaks
    lst = maps.regrid(a.df, a.df[drivers.TARGET_COL].to_numpy(), a.shape)
    return a, lst


def _input_sig(bbox, start, end, src, model):
    """Signature of the inputs that define an analysis — to detect when the
    displayed result is stale vs the current selection."""
    return (tuple(round(float(b), 4) for b in bbox),
            str(start), str(end), src, model)


cur_sig = _input_sig(bbox, start, end, src, model)
if run:
    with st.spinner(f"Running {src} analysis for {aoi_name} ..."):
        try:
            a, lst = analyze(tuple(bbox), aoi_name, start, end, src, model, project)
            st.session_state["result"] = (a, lst)
            st.session_state["result_sig"] = cur_sig
            # render the submission report once (not on every rerun)
            try:
                from src.report import figure_bytes
                st.session_state["report"] = (figure_bytes(a, a.cfg, "png"),
                                              figure_bytes(a, a.cfg, "pdf"),
                                              a.cfg.aoi_name)
            except Exception as e:
                st.session_state.pop("report", None)
                st.warning(f"Report render skipped: {e}")
        except Exception as e:
            st.error(f"Analysis failed: {e}")

if "result" in st.session_state:
    a, lst = st.session_state["result"]
    res, df, shape = a.res, a.df, a.shape
    stale = st.session_state.get("result_sig") != cur_sig

    if stale:
        st.warning(f"Showing the previous result for {a.cfg.aoi_name} "
                   f"({a.cfg.start} → {a.cfg.end}, {a.source}). "
                   "Run analysis to update for your current selection.")

    with right:
        st.subheader("Heat map")
        if a.source == "synthetic":
            st.info("Demo mode — synthetic data, not real imagery. Switch to "
                    "Live satellite (GEE) for this location.")
        ov = st.radio("Layer", ["Temperature", "Hotspots", "Plan"],
                      horizontal=True, key="ov_layer",
                      label_visibility="collapsed")
        oc1, oc2 = st.columns([2, 1])
        opacity = oc1.slider("Opacity", 0.0, 1.0, 0.70, 0.05, key="ov_opacity")
        base = oc2.selectbox("Base map", list(_BASE_TILES), key="ov_base")

        lst_range = None
        if ov == "Temperature":
            rgba, lst_range = maps.lst_rgba(lst)
        elif ov == "Hotspots":
            rgba = maps.hotspot_rgba(
                maps.regrid(df, df["hotspot"].to_numpy("float32"), shape))
        else:
            code = {s: i for i, s in enumerate(SCEN_COLORS)}
            cg = maps.regrid(a.plan,
                             a.plan["best_strategy"].map(code).to_numpy("float32"),
                             shape)
            rgba = maps.category_rgba(cg, {i: SCEN_COLORS[s]
                                           for s, i in code.items()})

        try:
            from streamlit_folium import st_folium
            st_folium(overlay_map(a.cfg.bbox, rgba, opacity, base), height=400,
                      use_container_width=True, key="result_map",
                      returned_objects=[])
        except Exception as e:
            st.warning(f"Map overlay unavailable: {e}")
            st.image(np.nan_to_num(rgba), use_container_width=True)

        if lst_range:
            st.markdown(colorbar_html(*lst_range), unsafe_allow_html=True)
        elif ov == "Plan":
            present = [s for s in SCEN_COLORS if s in set(a.plan["best_strategy"])]
            st.markdown(legend_html(present), unsafe_allow_html=True)
        else:
            st.caption(f"Cyan = hotspots · top {df['hotspot'].mean():.0%} of "
                       "cells by surface temperature.")

        m = res.metrics
        sp = m.get("r2_spatial")
        c1, c2 = st.columns(2)
        c1.metric("Honest R²", f"{sp:.3f}" if sp is not None else "—",
                  help="2×2 spatial cross-validation, no neighbour leakage — the "
                       "model's true skill on unseen ground. The number to quote.")
        c2.metric("In-scene R²", f"{m['r2']:.3f}",
                  help=f"Random-pixel hold-out · MAE {m['mae']:.2f} °C. "
                       "Optimistic — adjacent pixels leak across the split.")

        atm = getattr(a, "atmosphere", {}) or {}
        if atm:
            st.caption(
                f"ERA5 scene mean — air {atm.get('air_temp_C', '—')} °C · "
                f"humidity {atm.get('humidity_pct', '—')} % · "
                f"wind {atm.get('wind_m_s', '—')} m/s "
                "(near-uniform city-wide; context, not a per-pixel feature).")

        if "report" in st.session_state:
            png, pdf, rname = st.session_state["report"]
            d1, d2 = st.columns(2)
            d1.download_button("Report (PDF)", pdf, f"urban_heat_{rname}.pdf",
                               "application/pdf", use_container_width=True)
            d2.download_button("Report (PNG)", png, f"urban_heat_{rname}.png",
                               "image/png", use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Drivers", "Cooling", "Plan", "Validation"])

    with tab1:
        st.markdown("#### What drives the heat")
        st.caption("Each factor's total control over local surface temperature "
                   "(SHAP). Longer bar = bigger lever.")
        grp = group_importance(res.importance)
        gimp = pd.DataFrame({"driver": list(grp), "impact": list(grp.values())})
        st.altair_chart(hbar(gimp, "impact", "driver"), use_container_width=True)
        with st.expander(f"Full breakdown — all {len(res.importance)} features"):
            st.caption(
                "Bars above group these by physical factor. Each includes its "
                "base index, local texture (5-px std), neighbourhood means "
                "(≈210 m / 630 m — Landsat thermal is ~100 m, so surroundings "
                "dominate a pixel's LST), and ESA WorldCover class fractions.")
            imp = pd.DataFrame({"driver": [pretty(k) for k in res.importance],
                                "impact": list(res.importance.values())})
            st.altair_chart(hbar(imp, "impact", "driver"), use_container_width=True)
            st.caption(
                "Albedo reads low — confounded with built-up/vegetation — so "
                "cool-roof/pavement cooling uses a physics anchor "
                f"(~{BASE.lst_per_albedo:.0f} °C per +1.0 albedo), not this chart.")

    with tab2:
        st.markdown("#### Cooling potential by strategy")
        st.caption("Apply each intervention to eligible pixels, re-predict LST. "
                   "Bar = mean °C cooled where it applies.")
        rows = [{"strategy": _SCEN.get(k, k), "cooling_C": round(v.mean_cooling, 2),
                 "pixels": v.pixels} for k, v in a.scenarios.items()]
        sc = pd.DataFrame(rows).sort_values("cooling_C", ascending=False)
        st.altair_chart(hbar(sc, "cooling_C", "strategy", color="#5ec8a8"),
                        use_container_width=True)
        st.dataframe(
            sc.rename(columns={"cooling_C": "mean cooling °C",
                               "pixels": "eligible pixels"}),
            use_container_width=True, hide_index=True)

        st.markdown("##### Explore one strategy")
        pick = st.selectbox("Strategy", list(a.scenarios),
                            format_func=lambda k: _SCEN.get(k, k), key="scen_pick")
        sr = a.scenarios[pick]
        npix = int(sr.eligible.sum())
        e1, e2, e3 = st.columns(3)
        e1.metric("Mean cooling", f"{sr.mean_cooling:.2f} °C")
        e2.metric("Eligible area", f"{npix / max(len(df), 1):.0%} of cells")
        if npix:
            e3.metric("Best-case cell",
                      f"{np.nanmax(sr.delta_lst[sr.eligible]):.1f} °C")
        if npix:
            cg = maps.regrid(df, sr.delta_lst, shape)
            cg[~np.nan_to_num(maps.regrid(df, sr.eligible.astype("float32"),
                                          shape)).astype(bool)] = np.nan
            rgba, rng = maps.lst_rgba(cg, cmap="YlGnBu")
            try:
                from streamlit_folium import st_folium
                st_folium(overlay_map(a.cfg.bbox, rgba, 0.75, "Streets"),
                          height=340, use_container_width=True,
                          key="scen_map", returned_objects=[])
                st.caption(f"Per-cell cooling from **{_SCEN.get(pick, pick)}** "
                           f"where it applies (darker = more °C, "
                           f"{rng[0]:.1f}–{rng[1]:.1f} °C). Empty = not eligible.")
            except Exception:
                st.image(np.nan_to_num(rgba), use_container_width=True)
        else:
            st.info("No eligible cells for this strategy in this AOI.")

    with tab3:
        st.markdown("#### Where to act — best strategy per cell")
        st.caption("Each cell's single best strategy (roofs in the built core, "
                   "greening / water on open land). Priority cells rank highest "
                   "on cooling × population exposure.")
        plan = a.plan
        prio = plan[plan["priority"]] if "priority" in plan else plan
        k1, k2, k3 = st.columns(3)
        k1.metric("Most-recommended", _SCEN.get(plan["best_strategy"].mode().iat[0], "—"))
        k2.metric("Mean cooling (priority)", f"{prio['cooling_C'].mean():.2f} °C")
        k3.metric("Priority cells", f"{len(prio):,} / {len(plan):,}")

        present = [s for s in SCEN_COLORS if s in set(plan["best_strategy"])]
        st.caption("Switch the map layer above to **Plan** to see every cell's "
                   "strategy over the city.")
        st.markdown("**Legend** " + legend_html(present), unsafe_allow_html=True)
        mix = (plan["best_strategy"].value_counts(normalize=True)
               .rename(lambda s: _SCEN.get(s, s)))
        st.caption("Mix: " + " · ".join(f"{k} {v:.0%}" for k, v in mix.items()))

        show = (prio.head(100)
                .assign(strategy=prio.head(100)["best_strategy"]
                        .map(lambda s: _SCEN.get(s, s)))
                [["row", "col", "strategy", "cooling_C", "pop", "score"]]
                .rename(columns={"cooling_C": "cooling °C", "pop": "population",
                                 "score": "priority"}))
        pmin, pmax = float(show["priority"].min()), float(show["priority"].max())
        # ProgressColumn errors if min == max (e.g. uniform population) — use a
        # plain column in that case.
        prio_cfg = (st.column_config.ProgressColumn("priority", format="%.1f",
                                                    min_value=pmin, max_value=pmax)
                    if pmax > pmin else
                    st.column_config.NumberColumn("priority", format="%.1f"))
        st.dataframe(show, use_container_width=True, hide_index=True,
                     column_config={"priority": prio_cfg})
        st.download_button("Download full plan (CSV)", plan.to_csv(index=False),
                           "intervention_plan.csv", use_container_width=True)

    with tab4:
        st.markdown("#### Model validation")
        st.caption("Predicted vs observed LST on held-out pixels — points on the "
                   "dashed 1:1 line are accurate.")
        if res.eval:
            v1, v2, v3 = st.columns(3)
            v1.metric("In-scene R²", f"{m['r2']:.3f}")
            v2.metric("Honest R²", f"{sp:.3f}" if sp is not None else "—")
            v3.metric("Mean abs. error", f"{m['mae']:.2f} °C")
            st.altair_chart(validation_chart(res.eval), use_container_width=True)
        else:
            st.info("No held-out evaluation available for this model.")

        st.markdown("##### Cross-sensor LST validation (ECOSTRESS)")
        lv = getattr(a, "lst_validation", None)
        if lv:
            w1, w2, w3 = st.columns(3)
            w1.metric("Sensor agreement r", f"{lv['pearson_r']:.3f}",
                      help="Pearson correlation, Landsat vs ECOSTRESS LST")
            w2.metric("Cross-sensor MAE", f"{lv['mae_C']:.2f} °C")
            w3.metric("Bias (ECO − Landsat)", f"{lv['bias_C']:+.2f} °C")
            st.caption(f"Two independent satellites agree on the surface "
                       f"temperature over {lv['n_pixels']:,} overlapping pixels — "
                       "external confirmation that the modelled LST target is real.")
        else:
            st.caption("ECOSTRESS cross-check: enable `validate.ecostress` in "
                       "config (Live mode). Skipped when ISS coverage is absent "
                       "for the AOI/dates.")
else:
    with right:
        st.info("Set region + dates on the left, then **Run analysis**.")
