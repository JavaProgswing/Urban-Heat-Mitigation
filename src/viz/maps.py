"""Render heat maps + driver importance + scenario outputs."""
from __future__ import annotations
from pathlib import Path

import numpy as np


def regrid(df, values, shape):
    """Scatter a per-pixel value vector back to a 2-D raster using row/col."""
    grid = np.full(shape, np.nan, dtype="float32")
    grid[df["row"].to_numpy(), df["col"].to_numpy()] = values
    return grid


# --- RGBA overlays for the interactive (Leaflet) map ----------------------- #
# Each returns a float (H,W,4) RGBA array with alpha=0 where there is no data,
# so the base map shows through. folium.ImageOverlay maps it to the AOI bounds,
# which projects it geographically (no stretching of the raw raster).

def lst_rgba(grid, cmap="inferno"):
    """Colour an LST grid; transparent where NaN."""
    import matplotlib.cm as cm
    g = np.asarray(grid, "float32")
    finite = np.isfinite(g)
    lo = float(np.nanmin(g)) if finite.any() else 0.0
    hi = float(np.nanmax(g)) if finite.any() else 1.0
    norm = (np.nan_to_num(g, nan=lo) - lo) / (hi - lo + 1e-9)
    rgba = getattr(cm, cmap)(norm)                  # (H,W,4) float 0..1
    rgba[..., 3] = finite.astype("float32")
    return rgba, (lo, hi)


def hotspot_rgba(hot_grid, color=(0.0, 0.9, 1.0)):
    """Cyan-ish overlay only on hotspot cells; transparent elsewhere."""
    h = np.nan_to_num(np.asarray(hot_grid, "float32")) > 0.5
    rgba = np.zeros((*h.shape, 4), "float32")
    rgba[h, 0], rgba[h, 1], rgba[h, 2] = color
    rgba[..., 3] = h.astype("float32")
    return rgba


def category_rgba(grid, color_map: dict):
    """Colour an integer-coded grid (e.g. best strategy id) by a {code: hex}
    map; transparent where code < 0 / NaN."""
    import matplotlib.colors as mcolors
    g = np.nan_to_num(np.asarray(grid, "float32"), nan=-1.0)
    rgba = np.zeros((*g.shape, 4), "float32")
    for code, hexc in color_map.items():
        m = g == code
        if m.any():
            r, gr, b = mcolors.to_rgb(hexc)
            rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = r, gr, b, 1.0
    return rgba


def save_heatmap(grid, out: Path, title="LST (deg C)", cmap="inferno"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_driver_bar(importance: dict, out: Path,
                    title="Driver importance (|impact| on LST)"):
    import matplotlib.pyplot as plt
    names = list(importance)[::-1]
    vals = [importance[k] for k in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, vals, color="#c0392b")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_validation_scatter(y_true, y_pred, out: Path, metrics: dict | None = None):
    """Predicted vs observed LST scatter with 1:1 line — visual accuracy check."""
    import matplotlib.pyplot as plt
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3, color="#c0392b")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Observed LST (deg C)")
    ax.set_ylabel("Predicted LST (deg C)")
    title = "Predicted vs observed LST"
    if metrics:
        title += f"  (R2={metrics.get('r2', 0):.3f}, MAE={metrics.get('mae', 0):.2f})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_scenario_bar(results: dict, out: Path):
    """Mean cooling per strategy (deg C)."""
    import matplotlib.pyplot as plt
    items = sorted(results.items(), key=lambda kv: -kv[1].mean_cooling)
    names = [k for k, _ in items]
    vals = [r.mean_cooling for _, r in items]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color="#2980b9")
    ax.set_ylabel("Mean cooling (deg C)")
    ax.set_title("Cooling potential by strategy")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def folium_lst(df, lst_grid, bbox, out: Path):
    """Interactive Leaflet map of LST over the AOI bbox."""
    import folium
    from folium.raster_layers import ImageOverlay
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    minx, miny, maxx, maxy = bbox
    norm = mcolors.Normalize(np.nanmin(lst_grid), np.nanmax(lst_grid))
    rgba = cm.inferno(norm(np.nan_to_num(lst_grid)))
    m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2],
                   zoom_start=12, tiles="CartoDB positron")
    ImageOverlay(rgba, bounds=[[miny, minx], [maxy, maxx]],
                 opacity=0.6).add_to(m)
    m.save(str(out))
    return out
