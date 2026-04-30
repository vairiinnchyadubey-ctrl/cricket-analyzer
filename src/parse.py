"""Parse Cricsheet JSONs into flat match-level and delivery-level CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


def parse_match(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    info = data["info"]
    match_id = path.stem

    teams = info.get("teams", [])
    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    by = outcome.get("by", {})
    toss = info.get("toss", {})

    match_row = {
        "match_id": match_id,
        "season": str(info.get("season", "")).split("/")[0],
        "date": (info.get("dates") or [None])[0],
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1": teams[0] if len(teams) > 0 else None,
        "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": toss.get("winner"),
        "toss_decision": toss.get("decision"),
        "winner": winner,
        "win_by_runs": by.get("runs"),
        "win_by_wickets": by.get("wickets"),
        "result": outcome.get("result"),  # "tie", "no result", or None
        "method": outcome.get("method"),  # e.g., D/L
        "player_of_match": (info.get("player_of_match") or [None])[0],
    }

    deliveries: list[dict] = []
    for inning_idx, inning in enumerate(data.get("innings", []), start=1):
        batting_team = inning.get("team")
        for over in inning.get("overs", []):
            over_num = over.get("over")
            for ball_idx, d in enumerate(over.get("deliveries", []), start=1):
                runs = d.get("runs", {})
                wkts = d.get("wickets", [])
                deliveries.append({
                    "match_id": match_id,
                    "inning": inning_idx,
                    "batting_team": batting_team,
                    "over": over_num,
                    "ball": ball_idx,
                    "batter": d.get("batter"),
                    "bowler": d.get("bowler"),
                    "non_striker": d.get("non_striker"),
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),
                    "wicket": 1 if wkts else 0,
                    "wicket_kind": wkts[0]["kind"] if wkts else None,
                    "player_out": wkts[0]["player_out"] if wkts else None,
                })
    return match_row, deliveries


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSONs in {RAW_DIR}. Run src/scrape.py first.")

    matches, all_deliveries = [], []
    for fp in tqdm(files, desc="parsing"):
        m, dels = parse_match(fp)
        matches.append(m)
        all_deliveries.extend(dels)

    matches_df = pd.DataFrame(matches).sort_values("date").reset_index(drop=True)
    deliv_df = pd.DataFrame(all_deliveries)

    matches_df.to_csv(PROC_DIR / "matches.csv", index=False)
    deliv_df.to_csv(PROC_DIR / "deliveries.csv", index=False)
    print(f"Wrote {len(matches_df)} matches, {len(deliv_df)} deliveries to {PROC_DIR}")


if __name__ == "__main__":
    main()
