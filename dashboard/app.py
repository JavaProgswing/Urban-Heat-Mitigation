"""Urban Heat AI — interactive dashboard.

Pick any city/region by name (or draw a box on the map), pick a date window,
and run the live Earth Engine + XGBoost analysis: heat hotspots, drivers,
cooling scenarios, and optimized parcel placement.

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
from src.features import drivers, zones, parcels
from src.pipeline import run_analysis
from src.viz import maps
from src import insights

st.set_page_config(page_title="Urban Heat AI", layout="wide")

BASE = load_config()
try:
    _ = BASE.gee_project
    GEE_CONFIG_ERROR = None
except ValueError as exc:
    GEE_CONFIG_ERROR = str(exc)

st.markdown("""
<style>
  footer, #MainMenu {visibility:hidden;}
  .block-container {padding-top:2rem; padding-bottom:2.5rem; max-width:1380px;}
  h1,h2,h3,h4 {letter-spacing:-0.015em; font-weight:650;}
  [data-testid="stMetricValue"] {font-size:1.55rem; font-weight:650;}
  [data-testid="stMetricLabel"] p {color:#8b97a7; font-size:.82rem;}
  [data-testid="stMetric"] {background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.075);
    border-radius:11px;padding:.75rem .85rem;}
  [data-testid="stCaptionContainer"], .stCaption {color:#7b8794; font-size:.82rem; line-height:1.45;}
  [data-testid="stDataFrame"] {border:1px solid rgba(255,255,255,.075);border-radius:11px;overflow:hidden;}
  [data-testid="stExpander"] {border-color:rgba(255,255,255,.075);border-radius:11px;overflow:hidden;}
  iframe {border-radius:12px;border:1px solid rgba(255,255,255,.08) !important;}
  [data-testid="stSidebar"] {
    border-right:1px solid rgba(255,255,255,.08);
    background:#111925;
  }
  [data-testid="stSidebar"] .block-container {padding:1.15rem 1.15rem 1.5rem;}
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color:#c7d2df;font-size:.78rem;font-weight:600;letter-spacing:.01em;
  }
  [data-testid="stSidebar"] input,
  [data-testid="stSidebar"] [data-baseweb="select"] > div {
    border-radius:10px;
  }
  [data-testid="stSidebar"] .stButton > button {
    min-height:2.8rem;border-radius:10px;font-weight:700;
  }
  .side-brand {padding:.25rem .15rem 1rem;border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:1rem;}
  .side-eyebrow {color:#6db3f2;font-size:.68rem;font-weight:750;letter-spacing:.14em;text-transform:uppercase;}
  .side-title {color:#f4f8fc;font-size:1.3rem;font-weight:750;letter-spacing:-.02em;margin-top:.18rem;}
  .side-subtitle {color:#8392a5;font-size:.76rem;line-height:1.4;margin-top:.2rem;}
  .side-step {display:flex;align-items:center;gap:.65rem;margin:.9rem 0 .6rem;}
  .side-step-num {display:grid;place-items:center;width:1.45rem;height:1.45rem;border-radius:50%;
    color:#8ec8f7;background:rgba(109,179,242,.12);border:1px solid rgba(109,179,242,.28);
    font-size:.68rem;font-weight:750;}
  .side-step-title {color:#eef4fa;font-size:.88rem;font-weight:700;}
  .side-step-copy {color:#748397;font-size:.68rem;margin-top:-.05rem;}
  .main-hero {display:flex;align-items:center;justify-content:space-between;
    padding:.15rem 0 1rem;margin-bottom:.2rem;border-bottom:1px solid rgba(255,255,255,.065);}
  .main-title {font-size:1.75rem;line-height:1.15;font-weight:750;letter-spacing:-.025em;color:#f4f8fc;}
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
    "urban_greening": "Urban greening", "water_body": "New water features",
}
_SCEN_META = {
    "cool_roofs": ("Building roofs", "Reflective roof treatment on suitable buildings."),
    "cool_pavements": ("Road surfaces", "Higher-albedo treatment on mapped road segments."),
    "high_albedo_paint": ("Built surfaces", "Reflective coating on suitable built-up surfaces."),
    "green_roofs": ("Building roofs", "Vegetated roof systems where roof structure permits."),
    "urban_greening": ("Open land", "Trees and vegetation on feasible mapped open space."),
    "water_body": ("Open land", "Proposed blue-space on feasible open land; not existing water."),
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


def fmt_num(value) -> str:
    """Format visible dashboard measurements consistently."""
    try:
        return "—" if pd.isna(value) else f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


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
        '<div style="height:10px;display:grid;grid-template-columns:repeat(6,1fr);'
        'gap:2px;overflow:hidden;border-radius:4px">'
        '<span style="background:#000004"></span><span style="background:#3b0f70"></span>'
        '<span style="background:#8c2981"></span><span style="background:#de4968"></span>'
        '<span style="background:#fe9f6d"></span><span style="background:#fcfdbf"></span></div>'
        '<div style="display:flex;justify-content:space-between;color:#94a3b8;'
        f'font-size:12px;margin-top:4px"><span>{lo:.2f} °C · cooler</span>'
        f'<span>hotter · {hi:.2f} °C</span></div></div>'
    )


# distinct colour per intervention strategy (placement map + legend)
SCEN_COLORS = {
    "cool_roofs": "#4FC3F7", "cool_pavements": "#BA68C8",
    "high_albedo_paint": "#4DD0E1", "green_roofs": "#AED581",
    "urban_greening": "#66BB6A", "water_body": "#26A69A",
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
    '<div class="main-hero"><div class="main-title">Urban Heat Mitigation</div></div>',
    unsafe_allow_html=True)

# ----------------------------- sidebar: inputs ----------------------------- #
# One seamless AOI picker: search a place OR draw a rectangle on the map. No
# coordinate typing, no mode switch. A drawn box overrides the searched one
# until you search again or reset.
st.sidebar.markdown("""
<div class="side-brand">
  <div class="side-eyebrow">Climate intelligence</div>
  <div class="side-title">Urban Heat AI</div>
  <div class="side-subtitle">Satellite-to-parcel cooling decisions</div>
</div>
<div class="side-step">
  <span class="side-step-num">1</span>
  <div><div class="side-step-title">Choose an area</div>
  <div class="side-step-copy">Search or draw a custom boundary</div></div>
</div>
""", unsafe_allow_html=True)


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
    st.sidebar.success(f"Custom area · {w_km:.2f} × {h_km:.2f} km")
    if st.sidebar.button("Use searched place", use_container_width=True):
        st.session_state.pop("draw_override", None)
        st.rerun()
elif geo_bbox:
    bbox, aoi_name = geo_bbox, geo_name
    w_km, h_km = bbox_km(bbox)
    st.sidebar.success(f"{geo_name.replace('_', ' ').title()} · {w_km:.2f} × {h_km:.2f} km")
else:
    bbox, aoi_name = list(BASE.bbox), BASE.aoi_name

if min(bbox_km(bbox)) < 1.5:
    st.sidebar.warning("Small AOI (<1.5 km) — noisier R² with little spatial "
                       "variety. 2–5 km works best.")

st.sidebar.caption("Tip: draw a rectangle on the preview map for an exact study area.")
st.sidebar.markdown("""
<div class="side-step">
  <span class="side-step-num">2</span>
  <div><div class="side-step-title">Set the season</div>
  <div class="side-step-copy">Use a broad window for clear observations</div></div>
</div>
""", unsafe_allow_html=True)
today = dt.date.today()
dc1, dc2 = st.sidebar.columns(2)
start = dc1.date_input("Start", value=today - dt.timedelta(days=120))
end = dc2.date_input("End", value=today)
window_days = (end - start).days
date_valid = window_days > 0
if not date_valid:
    st.sidebar.error("End date must be after the start date.")
elif window_days < 30:
    st.sidebar.warning("Use at least 30 days to reduce cloud gaps.")
else:
    st.sidebar.caption(f"{window_days} days · combined across {BASE.lst_years} hot seasons")

st.sidebar.markdown("""
<div class="side-step">
  <span class="side-step-num">3</span>
  <div><div class="side-step-title">Run analysis</div>
  <div class="side-step-copy">Live imagery · constrained XGBoost</div></div>
</div>
""", unsafe_allow_html=True)
if GEE_CONFIG_ERROR is not None:
    st.sidebar.error(GEE_CONFIG_ERROR)

inputs_ready = date_valid and GEE_CONFIG_ERROR is None
run = st.sidebar.button("Analyze urban heat", type="primary",
                        use_container_width=True, disabled=not inputs_ready)
st.sidebar.caption("Landsat 8/9 · Sentinel-2 · GHSL · ERA5")

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


def parcel_plan_map(bbox, parcel_plan, opacity=0.75, base="Streets",
                    priority_only=True, cap=6000):
    """Leaflet vector map of actions on real OSM assets, not raster squares."""
    import folium
    import textwrap
    from branca.element import Element

    minx, miny, maxx, maxy = bbox
    span = max(abs(maxx - minx), abs(maxy - miny))
    zoom = 14 if span < 0.05 else 13 if span < 0.12 else 12 if span < 0.3 else 11
    tiles, attr = _BASE_TILES.get(base, _BASE_TILES["Streets"])
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2],
                   zoom_start=zoom, tiles=tiles, attr=attr)

    def short(value, width=32):
        try:
            if value is None or pd.isna(value):
                return "—"
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        return textwrap.shorten(text, width=width, placeholder="…") or "—"

    def degrees(value):
        try:
            return "—" if pd.isna(value) else f"{float(value):.2f} °C"
        except (TypeError, ValueError):
            return "—"

    # Leaflet controls normally sit above tooltip panes. Raise asset information
    # above controls and constrain its width so top-edge features remain readable.
    m.get_root().header.add_child(Element("""
    <style>
      .leaflet-tooltip-pane { z-index: 1200 !important; }
      .leaflet-popup-pane { z-index: 1300 !important; }
      .leaflet-tooltip.asset-tooltip, .leaflet-tooltip.water-tooltip {
        max-width: 290px; white-space: normal; line-height: 1.25; padding: 0;
        background: rgba(12,18,28,.96); color: #edf4fb;
        border: 1px solid rgba(255,255,255,.20); border-radius: 7px;
        box-shadow: 0 8px 28px rgba(0,0,0,.35);
      }
      .leaflet-tooltip.asset-tooltip table, .leaflet-tooltip.water-tooltip table {
        width: 280px; border-collapse: collapse; table-layout: fixed; margin: 0;
      }
      .leaflet-tooltip.asset-tooltip th, .leaflet-tooltip.asset-tooltip td,
      .leaflet-tooltip.water-tooltip th, .leaflet-tooltip.water-tooltip td {
        padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,.10);
        vertical-align: top; font-size: 12px; overflow-wrap: anywhere;
      }
      .leaflet-tooltip.asset-tooltip th, .leaflet-tooltip.water-tooltip th {
        width: 40%; color: #9eacba; font-weight: 600; text-align: left;
      }
      .leaflet-tooltip.asset-tooltip td, .leaflet-tooltip.water-tooltip td {
        color: #f2f6fa; font-weight: 600;
      }
      .leaflet-tooltip.asset-tooltip tr:last-child th,
      .leaflet-tooltip.asset-tooltip tr:last-child td,
      .leaflet-tooltip.water-tooltip tr:last-child th,
      .leaflet-tooltip.water-tooltip tr:last-child td { border-bottom: 0; }
      .leaflet-control-layers { border-radius: 8px; overflow: hidden; }
    </style>
    """))

    actions = parcel_plan.priority if priority_only else parcel_plan.actions
    actions = actions.head(cap).copy()
    if not actions.empty:
        asset_names = {"building": "Building", "road": "Road",
                       "open_land": "Open land"}
        action_names = {"cool_roofs": "Cool roof", "cool_pavements": "Cool pavement",
                        "high_albedo_paint": "Reflective paint",
                        "green_roofs": "Green roof", "urban_greening": "Greening",
                        "water_body": "New water feature"}

        actions["asset_label"] = actions["asset_type"].map(asset_names).fillna("Asset")
        actions["reference_label"] = actions["location_ref"].map(short)
        actions["action_label"] = actions["strategy"].map(action_names).fillna("Action")
        actions["lst_label"] = actions["surface_temp_C"].map(degrees)
        actions["cooling_label"] = actions["cooling_C"].map(degrees)
        actions["heat_label"] = actions["heat_excess_C"].map(degrees)
        action_layer = actions[["asset_type", "strategy", "asset_label",
                                "reference_label", "action_label", "lst_label",
                                "cooling_label", "heat_label", "geometry"]].copy()

        def style(feature):
            props = feature.get("properties", {})
            color = SCEN_COLORS.get(props.get("strategy"), "#94a3b8")
            if props.get("asset_type") == "road":
                return {"color": color, "weight": 3.0, "opacity": opacity,
                        "fillOpacity": 0.0}
            return {"color": color, "weight": 0.8, "opacity": min(1.0, opacity + 0.1),
                    "fillColor": color, "fillOpacity": opacity * 0.72}

        folium.GeoJson(
            action_layer.to_json(drop_id=True), name="Priority interventions",
            style_function=style,
            highlight_function=lambda _f: {"weight": 3, "fillOpacity": 0.90},
            tooltip=folium.GeoJsonTooltip(
                fields=["asset_label", "reference_label", "action_label",
                        "lst_label", "cooling_label", "heat_label"],
                aliases=["Asset", "Reference", "Action", "LST",
                         "Cooling", "Heat excess"],
                localize=False, sticky=True, class_name="asset-tooltip",
                direction="auto", opacity=0.98,
            ),
        ).add_to(m)

    # Draw reference water last so roads/buildings cannot cover narrow rivers.
    # Lines and polygons need different styling to remain legible at city scale.
    water = parcel_plan.context
    if water is not None and not water.empty:
        water = water.copy()
        water["water_name"] = water["name"].map(short)
        water["water_reference"] = water["location_ref"].map(short)
        water_layer = water[["water_name", "water_reference", "geometry"]].copy()

        def water_style(feature):
            geom = feature.get("geometry", {}).get("type", "")
            if "LineString" in geom:
                return {"color": "#168CE3", "weight": 3.2, "opacity": 0.95,
                        "fillOpacity": 0.0}
            return {"color": "#0D78C9", "weight": 1.8, "opacity": 0.95,
                    "fillColor": "#29A7F2", "fillOpacity": 0.34}

        folium.GeoJson(
            water_layer.to_json(drop_id=True),
            name=f"Existing water — reference only · {len(water):,}",
            style_function=water_style,
            highlight_function=lambda _f: {"color": "#7DD3FC", "weight": 4,
                                            "fillOpacity": 0.48},
            tooltip=folium.GeoJsonTooltip(
                fields=["water_name", "water_reference"],
                aliases=["Water feature", "Reference"],
                sticky=True, class_name="water-tooltip", direction="auto",
                opacity=0.98),
        ).add_to(m)
    folium.Rectangle([[miny, minx], [maxy, maxx]], color="#e74c3c",
                     weight=1, fill=False).add_to(m)
    # Bottom-left avoids the top-right area where asset tooltips commonly open.
    folium.LayerControl(collapsed=True, position="bottomleft").add_to(m)
    return m, len(actions)


def _drawn_bbox(out):
    """Pull a [minx,miny,maxx,maxy] from st_folium's last drawn rectangle."""
    feat = (out or {}).get("last_active_drawing")
    if not feat or feat.get("geometry", {}).get("type") != "Polygon":
        return None
    ring = feat["geometry"]["coordinates"][0]
    xs, ys = [p[0] for p in ring], [p[1] for p in ring]
    return [min(xs), min(ys), max(xs), max(ys)]


with st.expander("Region — search or draw your area on the map",
                 expanded="result" not in st.session_state):
    out = None
    try:
        from streamlit_folium import st_folium
        # Stable key keeps a freshly-drawn rectangle across reruns; restricted
        # return limits reruns to real draws (avoids the "draw resets" bug).
        out = st_folium(preview_map(bbox, draw=True), height=340,
                        use_container_width=True, key=map_key,
                        returned_objects=["last_active_drawing"])
    except Exception:
        st.map(pd.DataFrame({"lat": [(bbox[1] + bbox[3]) / 2],
                             "lon": [(bbox[0] + bbox[2]) / 2]}))
    nb = _drawn_bbox(out)
    if nb:
        cur = st.session_state.get("draw_override")
        if cur is None or any(abs(a - b) > 1e-5 for a, b in zip(nb, cur)):
            st.session_state["draw_override"] = nb
            st.rerun()
    st.caption(f"AOI{[round(b, 2) for b in bbox]} at {aoi_name} on "
               f"dates {start} → {end}")


# ------------------------------ run analysis ------------------------------- #
@st.cache_data(show_spinner=False)
def analyze(bbox, name, start, end):
    cfg = BASE.override(bbox=bbox, name=name, start=str(start), end=str(end))
    a = run_analysis(cfg, cv=False)
    # return plain data (cache-friendly) + keep model for live scenario tweaks
    lst = maps.regrid(a.df, a.df[drivers.TARGET_COL].to_numpy(), a.shape)
    return a, lst


def _input_sig(bbox, start, end):
    """Signature of the inputs that define an analysis — to detect when the
    displayed result is stale vs the current selection."""
    return (tuple(round(float(b), 4) for b in bbox),
            str(start), str(end))


cur_sig = _input_sig(bbox, start, end)
if run:
    with st.spinner(f"Running satellite analysis for {aoi_name} ..."):
        try:
            a, lst = analyze(tuple(bbox), aoi_name, start, end)
            st.session_state["result"] = (a, lst)
            st.session_state["result_sig"] = cur_sig
            # OSM is a placement/feasibility layer only. It is intentionally
            # computed after the thermal model and fails back to the grid map.
            st.session_state["parcel_plan"] = parcels.build_parcel_plan(
                a.cfg, a.df, a.shape, a.scenarios, budget_frac=0.30)
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
    parcel_plan = st.session_state.get("parcel_plan")
    # Also clean a ParcelPlan already held in Streamlit session state, so a hot
    # code reload fixes old `nan` labels without requiring a fresh OSM download.
    if parcel_plan is not None:
        parcel_plan.actions = parcels.display_names(parcel_plan.actions)
        parcel_plan.context = parcels.display_names(parcel_plan.context)
    stale = st.session_state.get("result_sig") != cur_sig

    if stale:
        st.warning(f"Showing the previous result for {a.cfg.aoi_name} "
                   f"({a.cfg.start} → {a.cfg.end}). "
                   "Run analysis to update for your current selection.")

    m = res.metrics
    sp = m.get("r2_spatial")
    # neighbourhood aggregation drives the labelled map + prioritisation panel
    try:
        ztbl, _zi, zsrc = zones.neighborhood_table(df, a.cfg.bbox, shape,
                                                   plan=a.plan)
    except Exception:
        ztbl, zsrc = pd.DataFrame(), "none"
    lst_rng = (float(np.nanmin(lst)), float(np.nanmax(lst)))

    o1, o2 = st.columns([3, 2], gap="large")
    with o1:
        st.subheader("Urban heat map")
        if len(ztbl):
            st.pyplot(maps.labeled_heatmap(lst, ztbl, lst_rng),
                      use_container_width=True)
            st.caption("Mean surface temperature per neighbourhood "
                       f"({'OSM localities' if zsrc == 'osm' else 'grid zones'}).")
        else:
            rgba, _ = maps.lst_rgba(lst)
            st.image(np.nan_to_num(rgba), use_container_width=True)
    with o2:
        st.subheader("Key findings")
        bullets = insights.generate_insights(a, ztbl, group_importance)
        st.markdown("\n".join(f"- {b}" for b in bullets) or "_No findings._")
        mc1, mc2 = st.columns(2)
        mc1.metric("Honest R²", f"{sp:.2f}" if sp is not None else "—",
                   help="2×2 spatial cross-validation, no leakage — true skill "
                        "on unseen ground. The number to quote.")
        mc2.metric("In-scene R²", f"{m['r2']:.2f}",
                   help=f"Random hold-out · MAE {m['mae']:.2f} °C (optimistic).")

    st.subheader("Neighbourhood prioritization")
    if len(ztbl):
        show = ztbl.assign(best=ztbl["top_strategy"].map(
            lambda s: _SCEN.get(s, s) if s else "—"))
        show = show[["zone", "mean_lst", "heat_risk", "pop_exposed", "best",
                     "priority"]].rename(columns={
                         "zone": "neighbourhood", "mean_lst": "temp °C",
                         "heat_risk": "heat risk", "pop_exposed": "population"})
        show = show.round(2)
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            column_config={"heat risk": st.column_config.ProgressColumn(
                "heat risk", format="%.2f", min_value=0, max_value=100)})
    else:
        st.caption("Zone aggregation unavailable for this AOI.")

    if "report" in st.session_state:
        png, pdf, rname = st.session_state["report"]
        d1, d2 = st.columns(2)
        d1.download_button("Report (PDF)", pdf, f"urban_heat_{rname}.pdf",
                           "application/pdf", use_container_width=True)
        d2.download_button("Report (PNG)", png, f"urban_heat_{rname}.png",
                           "image/png", use_container_width=True)

    st.divider()
    st.subheader("Interactive map")
    ov = st.radio("Layer", ["Temperature", "Hotspots", "Plan"], horizontal=True,
                  key="ov_layer", label_visibility="collapsed")
    oc1, oc2 = st.columns([2, 1])
    opacity = oc1.slider("Opacity", 0.0, 1.0, 0.70, 0.05, key="ov_opacity")
    base = oc2.selectbox("Base map", list(_BASE_TILES), key="ov_base")
    priority_only = (st.toggle("Priority assets only", value=True,
                               help="Show the highest-ranked 30% of OSM assets, "
                                    "not every technically eligible parcel.")
                     if ov == "Plan" and parcel_plan is not None and
                     not parcel_plan.actions.empty else True)
    lst_range = None
    rgba = None
    vector_plan = False
    if ov == "Temperature":
        rgba, lst_range = maps.lst_rgba(lst)
    elif ov == "Hotspots":
        rgba = maps.hotspot_rgba(
            maps.regrid(df, df["hotspot"].to_numpy("float32"), shape))
    else:
        vector_plan = (parcel_plan is not None and not parcel_plan.actions.empty)
        if not vector_plan:
            code = {s: i for i, s in enumerate(SCEN_COLORS)}
            cg = maps.regrid(a.plan,
                             a.plan["best_strategy"].map(code).to_numpy("float32"),
                             shape)
            rgba = maps.category_rgba(cg, {i: SCEN_COLORS[s]
                                           for s, i in code.items()})
    try:
        from streamlit_folium import st_folium
        if vector_plan:
            result_map, shown = parcel_plan_map(a.cfg.bbox, parcel_plan, opacity,
                                                base, priority_only)
        else:
            result_map, shown = overlay_map(a.cfg.bbox, rgba, opacity, base), None
        st_folium(result_map, height=400,
                  use_container_width=True, key="result_map", returned_objects=[])
    except Exception as e:
        st.warning(f"Map overlay unavailable: {e}")
        if rgba is not None:
            st.image(np.nan_to_num(rgba), use_container_width=True)
    if lst_range:
        st.markdown(colorbar_html(*lst_range), unsafe_allow_html=True)
    elif ov == "Plan":
        strategy_values = (set(parcel_plan.actions["strategy"]) if vector_plan
                           else set(a.plan["best_strategy"]))
        present = [s for s in SCEN_COLORS if s in strategy_values]
        st.markdown(legend_html(present), unsafe_allow_html=True)
        if vector_plan:
            scope = "priority" if priority_only else "eligible"
            water_count = (len(parcel_plan.context)
                           if parcel_plan.context is not None else 0)
            st.caption(f"Showing {shown:,} {scope} OSM assets. Buildings carry roof "
                       "actions, lines are road segments, and green/open polygons "
                       f"carry nature-based actions. Blue shows {water_count:,} existing "
                       "water features for reference; teal New water features are proposals.")
        elif parcel_plan is not None and parcel_plan.warning:
            st.warning(parcel_plan.warning)
    else:
        st.caption(f"Cyan = hotspots · top {df['hotspot'].mean():.2%} of cells.")
    atm = getattr(a, "atmosphere", {}) or {}
    if atm:
        st.caption(f"ERA5 scene mean — air {fmt_num(atm.get('air_temp_C'))} °C · "
                   f"humidity {fmt_num(atm.get('humidity_pct'))} % · "
                   f"wind {fmt_num(atm.get('wind_m_s'))} m/s (city-wide context).")

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
                f"(~{BASE.lst_per_albedo:.2f} °C per +1.00 albedo), not this chart.")

    with tab2:
        st.markdown("#### Cooling potential by strategy")
        st.caption("Apply each intervention to eligible pixels, re-predict LST. "
                   "Bar = mean °C cooled where it applies.")
        rows = [{"strategy_key": k, "strategy": _SCEN.get(k, k),
                 "cooling_C": round(v.mean_cooling, 2), "pixels": v.pixels}
                for k, v in a.scenarios.items()]
        sc = pd.DataFrame(rows).sort_values("cooling_C", ascending=False)
        st.altair_chart(hbar(sc, "cooling_C", "strategy", color="#5ec8a8"),
                        use_container_width=True)
        st.dataframe(
            sc.drop(columns="strategy_key").rename(
                columns={"cooling_C": "mean cooling °C",
                         "pixels": "eligible pixels"}),
            use_container_width=True, hide_index=True)

        st.markdown("##### Cost-effectiveness (cooling per ₹)")
        st.caption("A screening comparison using relative installation-cost tiers, "
                   "not a contractor quote. The best cooling per cost unit scores 100.")
        ce = insights.cost_effectiveness(a.scenarios)
        eligibility = sc.set_index("strategy")["pixels"] / max(len(df), 1)
        ce["eligible_share"] = ce["strategy"].map(eligibility).fillna(0.0)
        cec1, cec2 = st.columns([1.35, 1], gap="large")
        with cec1:
            st.altair_chart(hbar(ce, "value_index", "strategy", color=ACCENT),
                            use_container_width=True)
        with cec2:
            best_value = ce.iloc[0]
            st.metric("Best relative value", best_value["strategy"])
            st.caption(f"{best_value['cooling_C']:.2f} °C mean cooling · "
                       f"{best_value['cost']} relative cost · "
                       f"{best_value['eligible_share']:.2%} eligible coverage")
            st.dataframe(
                ce[["strategy", "cost", "value_index"]].round(2).rename(
                    columns={"cost": "relative cost", "value_index": "value index"}),
                use_container_width=True, hide_index=True,
                column_config={"value index": st.column_config.ProgressColumn(
                    "value index", format="%.2f", min_value=0, max_value=100)})

        with st.expander("Cool-material reference (literature)"):
            st.caption("Typical performance of cool surfaces — cost / durability "
                       "are not model outputs.")
            st.dataframe(insights.MATERIALS, use_container_width=True,
                         hide_index=True)

        st.divider()
        st.markdown("##### Explore a strategy")
        sx1, sx2 = st.columns([1, 1.7], gap="large")
        with sx1:
            pick = st.selectbox("Intervention", list(a.scenarios),
                                format_func=lambda k: _SCEN.get(k, k), key="scen_pick")
        with sx2:
            target, description = _SCEN_META.get(pick, ("Eligible surfaces", ""))
            st.markdown(f"**Best suited to:** {target}")
            st.caption(description)
            if pick == "water_body":
                st.caption("Existing rivers, lakes and canals are shown separately "
                           "as reference context and are never counted as proposed work.")
        sr = a.scenarios[pick]
        npix = int(sr.eligible.sum())
        applied = np.asarray(sr.delta_lst)[np.asarray(sr.eligible, dtype=bool)]
        p90 = float(np.nanpercentile(applied, 90)) if applied.size else np.nan
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Mean cooling", f"{sr.mean_cooling:.2f} °C")
        e2.metric("90th percentile", f"{p90:.2f} °C" if npix else "—")
        e3.metric("Eligible coverage", f"{npix / max(len(df), 1):.2%}")
        e4.metric("Eligible cells", f"{npix:,}")
        if npix:
            cg = maps.regrid(df, sr.delta_lst, shape)
            cg[~np.nan_to_num(maps.regrid(df, sr.eligible.astype("float32"),
                                          shape)).astype(bool)] = np.nan
            rgba, rng = maps.lst_rgba(cg, cmap="YlGnBu")
            try:
                from streamlit_folium import st_folium
                st_folium(overlay_map(a.cfg.bbox, rgba, opacity, base),
                          height=390, use_container_width=True,
                          key="scen_map", returned_objects=[])
                st.caption(f"Per-cell cooling from **{_SCEN.get(pick, pick)}** "
                           f"where feasible · {rng[0]:.2f}–{rng[1]:.2f} °C. "
                           "Unshaded areas are not eligible for this intervention.")
            except Exception:
                st.image(np.nan_to_num(rgba), use_container_width=True)
        else:
            st.info("No eligible cells for this strategy in this AOI.")

    with tab3:
        st.markdown("#### Where to act — feasible assets")
        st.caption("Temperature remains a coarse satellite/model field; placement "
                   "is snapped to real OSM buildings, road segments, and mapped "
                   "open land. Ranking combines heat, cooling and exposure.")
        plan = a.plan
        prio = plan[plan["priority"]] if "priority" in plan else plan
        parcel_actions = (parcel_plan.actions if parcel_plan is not None
                          else pd.DataFrame())
        parcel_prio = (parcel_plan.priority if parcel_plan is not None and
                       not parcel_actions.empty else pd.DataFrame())
        k1, k2, k3 = st.columns(3)
        if len(parcel_actions):
            k1.metric("Most-recommended",
                      _SCEN.get(parcel_actions["strategy"].mode().iat[0], "—"))
            k2.metric("Mean cooling (priority)",
                      f"{parcel_prio['cooling_C'].mean():.2f} °C")
            k3.metric("Priority assets",
                      f"{len(parcel_prio):,} / {len(parcel_actions):,}")
        else:
            k1.metric("Most-recommended", _SCEN.get(plan["best_strategy"].mode().iat[0], "—"))
            k2.metric("Mean cooling (priority)", f"{prio['cooling_C'].mean():.2f} °C")
            k3.metric("Priority cells", f"{len(prio):,} / {len(plan):,}")

        if len(parcel_actions):
            st.markdown("##### Intervention register")
            scope = st.radio("Scope", ["Priority assets", "All eligible assets"],
                             horizontal=True, key="plan_scope")
            register = (parcel_prio if scope == "Priority assets"
                        else parcel_actions).copy()
            fc1, fc2, fc3 = st.columns([1.4, 1, 1])
            with fc1:
                query = st.text_input("Search", placeholder="Road, park or nearby place",
                                      key="plan_search")
            with fc2:
                group_choice = st.selectbox(
                    "Group by", ["No grouping", "Asset type", "Strategy"],
                    key="plan_group")
            with fc3:
                sort_choice = st.selectbox(
                    "Sort by", ["Priority", "Cooling", "Surface temperature",
                                "Heat excess", "Reference"], key="plan_sort")

            type_options = sorted(register["asset_type"].dropna().unique())
            strategy_options = sorted(register["strategy"].dropna().unique())
            ff1, ff2, ff3 = st.columns([1, 1.4, .7])
            with ff1:
                selected_types = st.multiselect(
                    "Asset", type_options, default=type_options,
                    format_func=lambda v: {"building": "Building", "road": "Road",
                                           "open_land": "Open land"}.get(v, v),
                    key="plan_asset_filter")
            with ff2:
                selected_strategies = st.multiselect(
                    "Intervention", strategy_options, default=strategy_options,
                    format_func=lambda v: _SCEN.get(v, v), key="plan_strategy_filter")
            with ff3:
                ascending = st.checkbox("Ascending", value=False, key="plan_ascending")

            filtered = register[
                register["asset_type"].isin(selected_types) &
                register["strategy"].isin(selected_strategies)].copy()
            if query.strip():
                q = query.strip().lower()
                searchable = (filtered["name"].fillna("").astype(str) + " " +
                              filtered["location_ref"].fillna("").astype(str) + " " +
                              filtered["strategy"].map(_SCEN).fillna("")).str.lower()
                filtered = filtered[searchable.str.contains(q, regex=False)].copy()

            sort_columns = {"Priority": "priority_score", "Cooling": "cooling_C",
                            "Surface temperature": "surface_temp_C",
                            "Heat excess": "heat_excess_C", "Reference": "location_ref"}
            filtered = filtered.sort_values(sort_columns[sort_choice], ascending=ascending)

            if group_choice != "No grouping" and len(filtered):
                group_col = "asset_type" if group_choice == "Asset type" else "strategy"
                grouped = (filtered.groupby(group_col, dropna=False)
                           .agg(assets=("asset_id", "size"),
                                mean_cooling_C=("cooling_C", "mean"),
                                mean_heat_excess_C=("heat_excess_C", "mean"),
                                mean_priority=("priority_score", "mean"))
                           .reset_index())
                if group_col == "strategy":
                    grouped[group_col] = grouped[group_col].map(_SCEN)
                else:
                    grouped[group_col] = grouped[group_col].map(
                        {"building": "Building", "road": "Road", "open_land": "Open land"})
                grouped = grouped.rename(columns={group_col: group_choice.lower(),
                                                  "mean_cooling_C": "mean cooling °C",
                                                  "mean_heat_excess_C": "mean heat excess °C",
                                                  "mean_priority": "mean priority"}).round(2)
                st.dataframe(grouped, use_container_width=True, hide_index=True)

            if len(filtered):
                detail = (filtered.head(500).copy()
                          .assign(strategy=filtered.head(500)["strategy"]
                                  .map(lambda s: _SCEN.get(s, s)))
                          [["location_ref", "asset_type", "strategy", "surface_temp_C",
                            "cooling_C", "heat_excess_C", "priority_score", "map_url"]]
                          .rename(columns={"location_ref": "reference", "asset_type": "asset",
                                           "map_url": "open", "surface_temp_C": "LST °C",
                                           "cooling_C": "cooling °C",
                                           "heat_excess_C": "heat excess °C",
                                           "priority_score": "priority"}).round(2))
                st.caption(f"Showing {min(len(filtered), 500):,} of {len(filtered):,} matching assets.")
                pmin, pmax = float(detail["priority"].min()), float(detail["priority"].max())
                prio_cfg = (st.column_config.ProgressColumn(
                    "priority", format="%.2f", min_value=pmin, max_value=pmax)
                    if pmax > pmin else
                    st.column_config.NumberColumn("priority", format="%.2f"))
                st.dataframe(detail, use_container_width=True, hide_index=True,
                             column_config={"priority": prio_cfg,
                                            "open": st.column_config.LinkColumn(
                                                "open", display_text="View ↗")})
            else:
                st.info("No intervention assets match these filters.")
        else:
            show = (prio.head(100)
                    .assign(strategy=prio.head(100)["best_strategy"]
                            .map(lambda s: _SCEN.get(s, s)))
                    [["row", "col", "strategy", "cooling_C", "pop", "score"]]
                    .rename(columns={"cooling_C": "cooling °C", "pop": "population",
                                     "score": "priority"}))
            st.dataframe(show.round(2), use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download grid plan (CSV)", plan.to_csv(index=False),
                           "intervention_plan.csv", use_container_width=True)
        if len(parcel_actions):
            d2.download_button(
                "Download parcel plan (GeoJSON)",
                parcel_actions.to_json(drop_id=True), "parcel_intervention_plan.geojson",
                "application/geo+json", use_container_width=True)
        elif parcel_plan is not None and parcel_plan.warning:
            st.info(parcel_plan.warning)

    with tab4:
        st.markdown("#### Model validation")
        st.caption("Spatial holdout is the primary generalization check. Random "
                   "holdout is shown for reference and is usually more optimistic.")
        if res.eval:
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Spatial R²", f"{sp:.2f}" if sp is not None else "—",
                      help="Held-out geographic blocks; use this as the primary score.")
            v2.metric("Random R²", f"{m['r2']:.2f}",
                      help="Random pixel holdout; can benefit from spatial similarity.")
            v3.metric("Mean abs. error", f"{m['mae']:.2f} °C")
            v4.metric("RMSE", f"{m['rmse']:.2f} °C")
            yt = np.asarray(res.eval["y_true"], "float32")
            yp = np.asarray(res.eval["y_pred"], "float32")
            errors = yp - yt
            vc1, vc2 = st.columns([1.55, .75], gap="large")
            with vc1:
                st.altair_chart(validation_chart(res.eval), use_container_width=True)
            with vc2:
                st.markdown("##### Error profile")
                st.metric("Mean bias", f"{np.nanmean(errors):+.2f} °C")
                st.metric("90% within", f"±{np.nanpercentile(np.abs(errors), 90):.2f} °C")
                st.caption("Points close to the dashed 1:1 line have small prediction error.")
        else:
            st.info("No held-out evaluation available for this model.")

        lv = getattr(a, "lst_validation", None)
        with st.expander("External sensor check — ECOSTRESS", expanded=bool(lv)):
            if lv:
                w1, w2, w3 = st.columns(3)
                w1.metric("Sensor agreement r", f"{lv['pearson_r']:.2f}",
                          help="Pearson correlation, Landsat vs ECOSTRESS LST")
                w2.metric("Cross-sensor MAE", f"{lv['mae_C']:.2f} °C")
                w3.metric("Bias (ECO − Landsat)", f"{lv['bias_C']:+.2f} °C")
                st.caption(f"Independent ECOSTRESS observations overlap "
                           f"{lv['n_pixels']:,} pixels in this analysis.")
            else:
                st.caption("No usable ECOSTRESS overlap for this AOI and date window, "
                           "or the optional check is disabled in config.")
else:
    st.info("Search or draw a region above, set dates on the left, "
            "then **Run analysis**.")
