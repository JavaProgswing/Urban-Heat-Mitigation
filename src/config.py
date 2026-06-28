"""Load config.yaml + .env into a single typed config object."""
from __future__ import annotations
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
_GCP_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_PROJECT_PLACEHOLDERS = {
    "your-project-id", "your-gee-project-id", "gee-project-id", "project-id",
}


def validate_gee_project(value: Any) -> str:
    """Return a valid Google Cloud project id or raise an actionable error."""
    project = "" if value is None else str(value).strip()
    if not project or project.lower() in _PROJECT_PLACEHOLDERS:
        raise ValueError(
            "Google Earth Engine project is not configured. Set `gee.project` "
            "in config.yaml to the Google Cloud project registered for Earth Engine."
        )
    if not _GCP_PROJECT_RE.fullmatch(project):
        raise ValueError(
            f"Invalid Google Earth Engine project id in config.yaml: {project!r}. "
            "Use 6-30 lowercase letters, digits, or hyphens; start with a letter "
            "and end with a letter or digit."
        )
    return project


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = ROOT

    # convenience accessors -------------------------------------------------
    @property
    def bbox(self) -> list[float]:
        return self.raw["aoi"]["bbox"]

    @property
    def aoi_name(self) -> str:
        return self.raw["aoi"]["name"]

    @property
    def start(self) -> str:
        return self.raw["time"]["start"]

    @property
    def end(self) -> str:
        return self.raw["time"]["end"]

    @property
    def lst_years(self) -> int:
        """Years of the same season to stack for the LST composite (>=1)."""
        return max(1, int(self.raw["time"].get("lst_years", 1)))

    @property
    def resolution_m(self) -> int:
        return int(self.raw["project"]["resolution_m"])

    @property
    def validate_ecostress(self) -> bool:
        """Whether to fetch ECOSTRESS LST for cross-sensor validation."""
        return bool(self.raw.get("validate", {}).get("ecostress", False))

    @property
    def gee_project(self) -> str:
        return validate_gee_project(self.raw.get("gee", {}).get("project"))

    @property
    def scenarios(self) -> dict[str, dict]:
        return self.raw["scenarios"]

    @property
    def lst_per_albedo(self) -> float:
        """Physics prior: deg C surface cooling per +1.0 albedo (cool roofs)."""
        return float(self.raw.get("physics", {}).get("lst_per_albedo", 12.0))

    @property
    def lst_per_ndvi(self) -> float:
        """Physics prior: deg C cooling per +1.0 NDVI (greening / green roofs)."""
        return float(self.raw.get("physics", {}).get("lst_per_ndvi", 8.0))

    @property
    def lst_per_ndwi(self) -> float:
        """Physics prior: deg C cooling per +1.0 NDWI (water bodies)."""
        return float(self.raw.get("physics", {}).get("lst_per_ndwi", 6.0))

    @property
    def max_cooling_C(self) -> float:
        """Realism cap on a single intervention's area-averaged cooling."""
        return float(self.raw.get("physics", {}).get("max_cooling_C", 8.0))

    def path(self, key: str) -> Path:
        p = self.root / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def override(self, *, bbox=None, start=None, end=None,
                 name=None) -> "Config":
        """Return a copy with runtime AOI/date overrides (for the UI).
        Lets the dashboard analyze any region/dates without editing config.yaml.
        """
        raw = copy.deepcopy(self.raw)
        if bbox is not None:
            raw["aoi"]["bbox"] = [float(x) for x in bbox]
        if name is not None:
            raw["aoi"]["name"] = name
        if start is not None:
            raw["time"]["start"] = str(start)
        if end is not None:
            raw["time"]["end"] = str(end)
        return Config(raw=raw, root=self.root)


def load_config(path: str | Path = ROOT / "config.yaml") -> Config:
    load_dotenv(ROOT / ".env")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
