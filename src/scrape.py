"""Download IPL match JSONs from Cricsheet.

Cricsheet publishes a single zip per competition containing one JSON per match.
URL: https://cricsheet.org/downloads/ipl_json.zip
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS_KEEP = {"2023", "2024", "2025", "2026"}


def download_zip() -> bytes:
    print(f"Downloading {CRICSHEET_URL} ...")
    r = requests.get(CRICSHEET_URL, timeout=120, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="ipl_json.zip") as pbar:
        for chunk in r.iter_content(chunk_size=1 << 15):
            buf.write(chunk)
            pbar.update(len(chunk))
    return buf.getvalue()


def extract(zip_bytes: bytes) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    kept = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json")]
        for name in tqdm(names, desc="filtering matches"):
            with zf.open(name) as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            season = str(data.get("info", {}).get("season", "")).split("/")[0]
            if season not in SEASONS_KEEP:
                continue
            out = RAW_DIR / Path(name).name
            out.write_text(json.dumps(data))
            kept += 1
    return kept


def main() -> None:
    z = download_zip()
    n = extract(z)
    print(f"Saved {n} match files to {RAW_DIR}")


if __name__ == "__main__":
    main()
