"""Synthetic but physically-plausible driver rasters for offline demo/testing.

Lets the whole pipeline (model + scenarios + dashboard) run with NO GEE auth.
LST is generated from drivers via an energy-balance-like relation + noise, so
the ML model has a real signal to recover and scenarios produce sane deltas.
"""
from __future__ import annotations
import numpy as np

from .drivers import TARGET_COL


def make_grid(n: int = 128, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def smooth(scale=8.0):
        base = rng.standard_normal((n, n))
        # cheap smoothing via repeated box blur
        for _ in range(3):
            base = (base
                    + np.roll(base, 1, 0) + np.roll(base, -1, 0)
                    + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5.0
        return (base - base.min()) / (np.ptp(base) + 1e-9)

    ndvi = smooth() * 0.8 - 0.1          # -0.1..0.7
    build = smooth(4.0)                  # 0..1 built fraction
    ndbi = build * 0.6 - 0.1
    ndwi = np.clip(smooth() - 0.7, 0, 1) * 0.8
    albedo = np.clip(0.20 - 0.10 * ndvi + 0.05 * build, 0.05, 0.45)
    air_t = 32.0 + smooth(16.0) * 4.0    # 32..36 C background
    rh = 30.0 + smooth() * 30.0
    wind = 1.0 + smooth() * 3.0
    elev = 200.0 + smooth(12.0) * 50.0   # 200..250 m
    bld_h = np.clip(build * 18.0 + smooth(4.0) * 6.0 - 2.0, 0, 60)  # height (m)

    # distance-to-water (also fed to the LST formula); NDVI local texture
    from scipy import ndimage
    water = ndwi > 0.3
    water_dist = (ndimage.distance_transform_edt(~water).astype("float32")
                  if water.any() else np.full((n, n), float(n), "float32"))
    nd_mean = ndimage.uniform_filter(ndvi, size=5)
    ndvi_std = np.sqrt(np.maximum(
        ndimage.uniform_filter(ndvi * ndvi, size=5) - nd_mean ** 2, 0))

    # synthetic ESA-WorldCover-style class raster (codes per drivers.LULC_CLASSES)
    lulc = np.full((n, n), 30, dtype="float32")        # grass (default)
    lulc[(ndvi > 0.2) & (ndvi <= 0.5)] = 40            # crop
    lulc[ndvi > 0.5] = 10                              # tree
    lulc[(build <= 0.5) & (ndvi < 0.15)] = 60          # bare
    lulc[build > 0.5] = 50                             # built
    lulc[ndwi > 0.3] = 80                              # water (wins)

    # pseudo energy balance: built + low albedo + low veg + far-from-water +
    # low elevation -> hotter
    lst = (air_t
           + 12.0 * build
           - 8.0 * ndvi
           - 25.0 * (albedo - 0.20)
           - 6.0 * ndwi
           - 0.4 * wind
           - 0.02 * (elev - 200.0)
           + 0.05 * water_dist
           + np.random.default_rng(seed + 1).normal(0, 0.6, (n, n)))

    out = {
        "NDVI": ndvi.astype("float32"),
        "NDBI": ndbi.astype("float32"),
        "NDWI": ndwi.astype("float32"),
        "albedo": albedo.astype("float32"),
        "build_frac": build.astype("float32"),
        "BLD_H": bld_h.astype("float32"),
        "ELEV": elev.astype("float32"),
        "WATER_DIST": water_dist.astype("float32"),
        "NDVI_STD": ndvi_std.astype("float32"),
        "AIR_T": air_t.astype("float32"),
        "RH": rh.astype("float32"),
        "WIND": wind.astype("float32"),
        "LULC": lulc,
        TARGET_COL: lst.astype("float32"),
    }
    # Run the same derive_drivers as the live GEE path so the offline demo and
    # tests carry the full schema (neighbourhood _N context + LULC one-hots).
    from ..data.align import derive_drivers
    return derive_drivers(out)
