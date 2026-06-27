"""Offline test of raster alignment (no Earth Engine).

Writes synthetic GeoTIFFs at DIFFERENT resolutions, then checks align_stack
reproject-matches them all onto the reference grid. Validates the real-data
bridge without needing GEE auth. Skips cleanly if rasterio is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("rioxarray")
from rasterio.transform import from_bounds  # noqa: E402

from src.data.align import align_stack, derive_drivers, BAND_LAYOUT  # noqa: E402

BBOX = (77.0, 28.5, 77.2, 28.7)   # minx, miny, maxx, maxy


def _write(path, n, nbands, seed):
    rng = np.random.default_rng(seed)
    data = rng.random((nbands, n, n)).astype("float32")
    transform = from_bounds(*BBOX, n, n)
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=nbands,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)
    return path


def test_align_to_reference_grid(tmp_path):
    # reference Landsat LST 64x64; others at different resolutions
    paths = {
        "landsat": _write(tmp_path / "lst.tif", 64, 1, 1),
        "sentinel": _write(tmp_path / "s2.tif", 80, len(BAND_LAYOUT["sentinel"]), 2),
        "era5": _write(tmp_path / "era5.tif", 8, 3, 3),     # coarse met
        "ghsl": _write(tmp_path / "ghsl.tif", 100, 2, 4),
    }
    bands = align_stack({k: str(v) for k, v in paths.items()}, ref="landsat")
    ref_shape = bands["LST"].shape
    assert ref_shape == (64, 64)
    for name in ("NDVI", "NDBI", "NDWI", "AIR_T", "RH", "WIND", "BUILT", "POP"):
        assert bands[name].shape == ref_shape, f"{name} not aligned"


def test_derive_drivers_adds_albedo_buildfrac(tmp_path):
    paths = {
        "landsat": _write(tmp_path / "lst.tif", 32, 1, 1),
        "sentinel": _write(tmp_path / "s2.tif", 32, len(BAND_LAYOUT["sentinel"]), 2),
        "ghsl": _write(tmp_path / "ghsl.tif", 32, 2, 4),
    }
    stack = derive_drivers(align_stack({k: str(v) for k, v in paths.items()}))
    assert "albedo" in stack and "build_frac" in stack
    assert np.all((stack["build_frac"] >= 0) & (stack["build_frac"] <= 1))


if __name__ == "__main__":
    import tempfile
    # ignore_cleanup_errors: GDAL can hold file handles on Windows at rmtree
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        test_align_to_reference_grid(Path(d))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        test_derive_drivers_adds_albedo_buildfrac(Path(d))
    print("alignment tests passed")
