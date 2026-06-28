"""Cross-platform Google Earth Engine authentication.

Default mode = 'notebook': prints a URL, you open it, log in, copy the code,
paste it back here. No gcloud CLI needed (avoids the WinError 193 you hit when
the default 'gcloud' mode can't find/run gcloud).

    python scripts/gee_auth.py                       # notebook flow (robust)
    python scripts/gee_auth.py --mode localhost      # auto browser+redirect

Prereq: a Google account registered for Earth Engine + a Cloud project id
(free for research/non-commercial at https://earthengine.google.com).
Set `gee.project` in config.yaml before running this command.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="localhost",
                    choices=["localhost", "notebook", "gcloud"],
                    help="auth flow; 'localhost' auto-captures code in browser "
                         "(no paste); 'notebook' = manual paste. Neither needs "
                         "gcloud")
    args = ap.parse_args()
    try:
        project = load_config().gee_project
    except ValueError as exc:
        ap.error(str(exc))

    import ee
    print(f"Earth Engine authentication (mode={args.mode}) ...")
    ee.Authenticate(auth_mode=args.mode)     # notebook = paste-code, no gcloud
    print("Authenticated. Token saved to your user profile.")

    ee.Initialize(project=project)
    n = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").limit(1).size().getInfo()
    print(f"Connected to the configured project (test query OK, n={n}). "
          f"Ready for: python scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
