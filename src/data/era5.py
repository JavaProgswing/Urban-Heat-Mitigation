"""ERA5-Land meteorology (air temp, dewpoint->RH, wind speed).

Two backends:
  - GEE: ECMWF/ERA5_LAND/HOURLY (fast, no download) -> mean over time window.
  - CDS: cdsapi NetCDF download (for offline / CPCB fusion).
CPCB station CSVs (if available) can be merged in build_station_frame().
"""
from __future__ import annotations
from pathlib import Path
import math

from ..config import Config
from .gee_init import init_ee, aoi_geometry, safe_scale


def era5_means(cfg: Config):
    """Mean 2 m air temp (C), relative humidity (%), wind speed (m/s) image."""
    ee = init_ee(cfg.gee_project)
    aoi = aoi_geometry(cfg)
    col = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterBounds(aoi)
        .filterDate(cfg.start, cfg.end)
        .select(["temperature_2m", "dewpoint_temperature_2m",
                 "u_component_of_wind_10m", "v_component_of_wind_10m"])
        .mean()
    )
    t2m = col.select("temperature_2m").subtract(273.15).rename("AIR_T")
    d2m = col.select("dewpoint_temperature_2m").subtract(273.15)
    # Magnus RH
    rh = (
        d2m.expression(
            "100 * exp((17.625*Td)/(243.04+Td)) / exp((17.625*T)/(243.04+T))",
            {"Td": d2m, "T": t2m},
        ).rename("RH")
    )
    wind = (
        col.select("u_component_of_wind_10m").pow(2)
        .add(col.select("v_component_of_wind_10m").pow(2))
        .sqrt().rename("WIND")
    )
    return ee.Image.cat([t2m, rh, wind]).clip(aoi)


def export(cfg: Config, out: Path | None = None) -> Path:
    """Export ERA5 mean AIR_T/RH/WIND GeoTIFF (bands in that order) via geemap."""
    import geemap
    out = out or (cfg.path("raw") / f"era5_{cfg.aoi_name}.tif")
    geemap.ee_export_image(
        era5_means(cfg), filename=str(out),
        scale=safe_scale(cfg, 3), region=aoi_geometry(cfg), file_per_band=False,
    )
    return out


def download_cds(cfg: Config, out: Path | None = None) -> Path:
    """Download ERA5-Land NetCDF via Copernicus CDS API."""
    import cdsapi
    out = out or (cfg.path("raw") / "era5_land.nc")
    c = cdsapi.Client()
    minx, miny, maxx, maxy = cfg.bbox
    c.retrieve(
        cfg.raw["era5"]["cds_dataset"],
        {
            "variable": cfg.raw["era5"]["variables"],
            "area": [maxy, minx, miny, maxx],   # N,W,S,E
            "date": f"{cfg.start}/{cfg.end}",
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "format": "netcdf",
        },
        str(out),
    )
    return out
