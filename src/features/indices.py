"""Spectral + physical indices used as heat drivers."""
from __future__ import annotations
import numpy as np


def ndvi(nir, red):
    return _nd(nir, red)


def ndbi(swir, nir):
    return _nd(swir, nir)


def ndwi(green, nir):
    return _nd(green, nir)


def albedo_liang(b2, b4, b5, b6, b7):
    """Liang (2001) shortwave broadband albedo from Landsat-like bands."""
    return (0.356 * b2 + 0.130 * b4 + 0.373 * b5
            + 0.085 * b6 + 0.072 * b7 - 0.0018)


def albedo_from_ndvi(ndvi_arr):
    """Fallback albedo proxy when only NDVI is available.

    Vegetated surfaces ~0.15-0.20, bare/built ~0.10-0.30. Coarse linear map.
    """
    return np.clip(0.20 - 0.10 * np.asarray(ndvi_arr), 0.05, 0.45)


def _nd(a, b):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = a + b
    out = np.where(denom == 0, 0.0, (a - b) / denom)
    return out.astype("float32")
