"""Cross-platform Google Earth Engine authentication.

Default mode = 'notebook': prints a URL, you open it, log in, copy the code,
paste it back here. No gcloud CLI needed (avoids the WinError 193 you hit when
the default 'gcloud' mode can't find/run gcloud).

    python scripts/gee_auth.py                       # notebook flow (robust)
    python scripts/gee_auth.py --mode localhost      # auto browser+redirect
    python scripts/gee_auth.py --project my-proj-id  # also verify connection

Prereq: a Google account registered for Earth Engine + a Cloud project id
(free for research/non-commercial at https://earthengine.google.com).
After this, set `gee.project` in config.yaml (or GEE_PROJECT_ID in .env),
then run:  python scripts/run_pipeline.py --source gee
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="localhost",
                    choices=["localhost", "notebook", "gcloud"],
                    help="auth flow; 'localhost' auto-captures code in browser "
                         "(no paste); 'notebook' = manual paste. Neither needs "
                         "gcloud")
    ap.add_argument("--project", default=None,
                    help="GEE Cloud project id to verify initialization")
    args = ap.parse_args()

    import ee
    print(f"Earth Engine authentication (mode={args.mode}) ...")
    ee.Authenticate(auth_mode=args.mode)     # notebook = paste-code, no gcloud
    print("Authenticated. Token saved to your user profile.")

    if args.project:
        ee.Initialize(project=args.project)
        n = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").limit(1).size().getInfo()
        print(f"Connected to project '{args.project}' (test query OK, n={n}). "
              f"Ready for: python scripts/run_pipeline.py --source gee")


if __name__ == "__main__":
    main()
