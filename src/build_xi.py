"""Extract playing XIs from raw Cricsheet JSONs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def main() -> None:
    rows = []
    for fp in tqdm(sorted(RAW.glob("*.json")), desc="parsing XIs"):
        data = json.loads(fp.read_text())
        info = data.get("info", {})
        players = info.get("players", {}) or {}
        for team, names in players.items():
            for name in names:
                rows.append({"match_id": fp.stem, "team": team, "player": name})
    df = pd.DataFrame(rows)
    out = PROC / "xi.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} player-rows for {df['match_id'].nunique()} matches → {out}")


if __name__ == "__main__":
    main()
