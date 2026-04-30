"""Per-match venue pitch-character features using rolling history.

For each match, we compute (using ONLY past matches at the same venue):
  - venue_avg_1st_inn  — pitch high/low scoring
  - venue_avg_total    — total runs per match
  - venue_runs_per_wkt — pace of run flow (high = batting deck)
  - venue_sixes_per_match — six-friendly?
  - venue_bat_first_winrate — does posting first win here?
  - venue_pp_er         — Powerplay run rate trend (early swing/dew)
  - venue_death_er      — Death-overs run rate (collapse-prone or chase haven?)

Output: data/processed/venue_features.csv  (one row per match_id)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

WINDOW = 8  # last K venue matches used


def main() -> None:
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    matches = matches[matches["winner"].notna()].sort_values("date").reset_index(drop=True)
    deliveries = pd.read_csv(PROC / "deliveries.csv")

    # Pre-aggregate per match
    inn = deliveries.groupby(["match_id", "inning"]).agg(
        runs=("runs_total", "sum"),
        balls=("runs_total", "count"),
        wkts=("wicket", "sum"),
    ).reset_index()
    first_inn = inn[inn["inning"] == 1].groupby("match_id")["runs"].first()
    total_runs = deliveries.groupby("match_id")["runs_total"].sum()
    total_wkts = deliveries.groupby("match_id")["wicket"].sum()
    sixes = deliveries.groupby("match_id")["runs_batter"].apply(lambda s: (s == 6).sum())
    pp = deliveries[deliveries["over"] < 6].groupby("match_id").agg(
        pp_runs=("runs_total", "sum"), pp_balls=("runs_total", "count")
    )
    de = deliveries[deliveries["over"] >= 15].groupby("match_id").agg(
        de_runs=("runs_total", "sum"), de_balls=("runs_total", "count")
    )

    # Bat-first win flag per match
    bat_first = []
    for _, m in matches.iterrows():
        # The team that batted in inning 1 = first batting team
        first_bat = deliveries[(deliveries["match_id"] == m["match_id"]) & (deliveries["inning"] == 1)]["batting_team"]
        won = m["winner"] == first_bat.iloc[0] if not first_bat.empty else False
        bat_first.append(int(won))
    matches["bat_first_won"] = bat_first

    # Venue rolling state
    venue_history: dict[str, list[dict]] = {}

    rows = []
    for _, m in tqdm(matches.iterrows(), total=len(matches), desc="venue features"):
        v = m["venue"]; mid = m["match_id"]
        prior = venue_history.get(v, [])[-WINDOW:]

        if prior:
            avg_1st = np.mean([p["first_inn"] for p in prior if pd.notna(p["first_inn"])])
            avg_total = np.mean([p["total"] for p in prior])
            tot_runs = np.sum([p["total"] for p in prior])
            tot_wkts = np.sum([p["wkts"] for p in prior])
            avg_sixes = np.mean([p["sixes"] for p in prior])
            bat_winrate = np.mean([p["bat_first_won"] for p in prior])
            tot_pp_runs = np.sum([p["pp_runs"] for p in prior])
            tot_pp_balls = np.sum([p["pp_balls"] for p in prior])
            tot_de_runs = np.sum([p["de_runs"] for p in prior])
            tot_de_balls = np.sum([p["de_balls"] for p in prior])
            pp_er = (tot_pp_runs / (tot_pp_balls / 6)) if tot_pp_balls else 8.0
            de_er = (tot_de_runs / (tot_de_balls / 6)) if tot_de_balls else 9.0
            rpw = tot_runs / max(tot_wkts, 1)
        else:
            # Defaults
            avg_1st, avg_total, avg_sixes, bat_winrate = 165.0, 320.0, 14.0, 0.5
            pp_er, de_er, rpw = 8.0, 9.0, 23.0

        rows.append({
            "match_id": mid,
            "venue_avg_1st_inn": float(avg_1st),
            "venue_avg_total": float(avg_total),
            "venue_runs_per_wkt": float(rpw),
            "venue_sixes_per_match": float(avg_sixes),
            "venue_bat_first_winrate": float(bat_winrate),
            "venue_pp_er": float(pp_er),
            "venue_death_er": float(de_er),
        })

        # Update venue history AFTER processing this match
        venue_history.setdefault(v, []).append({
            "first_inn": first_inn.get(mid, np.nan),
            "total": total_runs.get(mid, 0),
            "wkts": total_wkts.get(mid, 0),
            "sixes": sixes.get(mid, 0),
            "bat_first_won": int(m["bat_first_won"]),
            "pp_runs": pp.loc[mid, "pp_runs"] if mid in pp.index else 0,
            "pp_balls": pp.loc[mid, "pp_balls"] if mid in pp.index else 0,
            "de_runs": de.loc[mid, "de_runs"] if mid in de.index else 0,
            "de_balls": de.loc[mid, "de_balls"] if mid in de.index else 0,
        })

    df = pd.DataFrame(rows)
    out = PROC / "venue_features.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} venue feature rows → {out}")


if __name__ == "__main__":
    main()
