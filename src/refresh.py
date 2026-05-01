"""Daily refresh — keep a rolling 2-year window of IPL data and rebuild everything.

Run this once a day (cron, launchd, or manual).

Steps:
  1. Re-download Cricsheet ipl_json.zip
  2. Keep only matches whose date is within the last 2 years (older ones get dropped)
  3. Re-parse to matches.csv / deliveries.csv
  4. Rebuild all features (form, venue pitch, kitchen-sink, external/weather)
  5. Rebuild the website index.html

You don't have to scrape any other source — Cricsheet's zip is the canonical feed
and is small (~4MB).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"

STEPS = [
    ("Download + filter (rolling 2-year window)", [str(PY), "-m", "src.scrape"]),
    ("Parse matches & deliveries",                 [str(PY), "-m", "src.parse"]),
    ("Build playing XIs",                          [str(PY), "-m", "src.build_xi"]),
    ("Build core features",                        [str(PY), "-m", "src.features"]),
    ("Build venue pitch features",                 [str(PY), "-m", "src.venue_features"]),
    ("Build kitchen-sink features",                [str(PY), "-m", "src.kitchen_sink"]),
    ("Fetch weather + venue attrs (cached)",       [str(PY), "-m", "src.external_features"]),
    ("Rebuild website",                            [str(PY), "-m", "src.build_site"]),
]


def main() -> None:
    for label, cmd in STEPS:
        print(f"\n→ {label}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED at: {label}")
            sys.exit(result.returncode)
    print("\n✓ Refresh complete. Deploy by copying web/index.html to docs/index.html and pushing to git.")


if __name__ == "__main__":
    main()
