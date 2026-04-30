"""Build per-match playing-XI strength features.

For each match:
  - Take each team's 11 players
  - For each player, compute rolling stats from the last K matches *before* this match
  - Aggregate XI -> team strength numbers
  - Output features csv with match_id + xi-derived numeric columns
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ROLLING_INNINGS = 8


def _player_match_stats(deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-(match, player) batting and bowling stats."""
    # Batting per match per batter
    bat = deliveries.groupby(["match_id", "batter"]).agg(
        runs=("runs_batter", "sum"),
        balls=("runs_batter", "count"),
    ).reset_index()
    bat.columns = ["match_id", "player", "runs", "balls"]

    # Bowling per match per bowler
    bowl = deliveries.groupby(["match_id", "bowler"]).agg(
        runs_conceded=("runs_total", "sum"),
        balls=("runs_total", "count"),
        wickets=("wicket", "sum"),
    ).reset_index()
    bowl.columns = ["match_id", "player", "runs_conceded", "balls_bowled", "wickets"]
    return bat, bowl


def _rolling_player_stats(player_match: pd.DataFrame, key: str, k: int = ROLLING_INNINGS) -> dict:
    """Build dict: player -> ordered list of (match_id, stats dict). Rolling state per player."""
    # We'll just return the raw frame sorted; lookup is per (player, match_id) prior-state.
    return player_match


def build() -> pd.DataFrame:
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    matches = matches[matches["winner"].notna()].sort_values("date").reset_index(drop=True)
    deliveries = pd.read_csv(PROC / "deliveries.csv")
    xi = pd.read_csv(PROC / "xi.csv")

    bat_pm, bowl_pm = _player_match_stats(deliveries)

    # Add date for ordering
    bat_pm = bat_pm.merge(matches[["match_id", "date"]], on="match_id", how="left").sort_values("date")
    bowl_pm = bowl_pm.merge(matches[["match_id", "date"]], on="match_id", how="left").sort_values("date")

    # Group per player for fast rolling lookup
    bat_groups = {p: g.reset_index(drop=True) for p, g in bat_pm.groupby("player")}
    bowl_groups = {p: g.reset_index(drop=True) for p, g in bowl_pm.groupby("player")}

    def player_bat_state(player: str, before_date) -> dict:
        g = bat_groups.get(player)
        if g is None:
            return None
        prior = g[g["date"] < before_date].tail(ROLLING_INNINGS)
        if prior.empty:
            return None
        runs = prior["runs"].sum(); balls = prior["balls"].sum(); innings = len(prior)
        return {
            "avg": runs / innings,
            "sr": (runs / balls * 100) if balls else 0,
            "innings": innings,
        }

    def player_bowl_state(player: str, before_date) -> dict:
        g = bowl_groups.get(player)
        if g is None:
            return None
        prior = g[g["date"] < before_date].tail(ROLLING_INNINGS)
        if prior.empty:
            return None
        runs = prior["runs_conceded"].sum(); balls = prior["balls_bowled"].sum(); wkts = prior["wickets"].sum(); innings = len(prior)
        return {
            "wpm": wkts / innings,
            "er": (runs / (balls / 6)) if balls else 0,
            "innings": innings,
        }

    rows = []
    xi_by_match = xi.groupby(["match_id", "team"])["player"].apply(list).to_dict()

    for _, m in tqdm(matches.iterrows(), total=len(matches), desc="XI features"):
        mid, t1, t2, d = m["match_id"], m["team1"], m["team2"], m["date"]
        xi1 = xi_by_match.get((mid, t1), [])
        xi2 = xi_by_match.get((mid, t2), [])

        def aggregate(team_xi, before_date):
            bat_avgs, bat_srs, bowl_wpms, bowl_ers = [], [], [], []
            for p in team_xi:
                bs = player_bat_state(p, before_date)
                if bs and bs["innings"] >= 3:
                    bat_avgs.append(bs["avg"]); bat_srs.append(bs["sr"])
                ws = player_bowl_state(p, before_date)
                if ws and ws["innings"] >= 3:
                    bowl_wpms.append(ws["wpm"]); bowl_ers.append(ws["er"])

            # Take top-5 batters (by avg) and top-4 bowlers (by wkts/match)
            bat_avgs = sorted(bat_avgs, reverse=True)[:5]
            bat_srs = sorted(bat_srs, reverse=True)[:5]
            bowl_wpms = sorted(bowl_wpms, reverse=True)[:4]
            bowl_ers = sorted(bowl_ers)[:4]
            return {
                "bat_avg": np.mean(bat_avgs) if bat_avgs else 25.0,
                "bat_sr": np.mean(bat_srs) if bat_srs else 130.0,
                "bowl_wpm": np.mean(bowl_wpms) if bowl_wpms else 1.0,
                "bowl_er": np.mean(bowl_ers) if bowl_ers else 9.0,
            }

        a1, a2 = aggregate(xi1, d), aggregate(xi2, d)
        rows.append({
            "match_id": mid,
            "t1_xi_bat_avg": a1["bat_avg"], "t2_xi_bat_avg": a2["bat_avg"],
            "t1_xi_bat_sr":  a1["bat_sr"],  "t2_xi_bat_sr":  a2["bat_sr"],
            "t1_xi_bowl_wpm":a1["bowl_wpm"],"t2_xi_bowl_wpm":a2["bowl_wpm"],
            "t1_xi_bowl_er": a1["bowl_er"], "t2_xi_bowl_er": a2["bowl_er"],
            "xi_bat_diff": a1["bat_avg"] - a2["bat_avg"],
            "xi_sr_diff":  a1["bat_sr"]  - a2["bat_sr"],
            "xi_bowl_diff": a1["bowl_wpm"] - a2["bowl_wpm"],
            "xi_er_diff":  a2["bowl_er"] - a1["bowl_er"],  # lower-is-better, flipped
        })

    df = pd.DataFrame(rows)
    out = PROC / "xi_features.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} XI feature rows → {out}")
    return df


if __name__ == "__main__":
    build()
