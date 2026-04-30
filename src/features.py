"""Build per-match features for winner prediction.

Each match becomes one row. Target = 1 if team1 wins, 0 if team2 wins.
Features are computed using ONLY data from prior matches (no leakage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"

RECENT_N = 5    # rolling form window
TOSS_N = 10     # rolling toss-win window


def _team_run_rate(deliveries: pd.DataFrame) -> pd.DataFrame:
    scored = deliveries.groupby(["match_id", "batting_team"])["runs_total"].sum().reset_index()
    scored.columns = ["match_id", "team", "runs_scored"]
    return scored


def _team_balls_faced(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Balls faced per match per team (used for NRR overs)."""
    bf = deliveries.groupby(["match_id", "batting_team"]).size().reset_index(name="balls_faced")
    bf.columns = ["match_id", "team", "balls_faced"]
    return bf


def _season_table_snapshot(season_state: dict) -> dict:
    """Compute current league rank for each team in the season."""
    teams = list(season_state.keys())
    standings = []
    for team in teams:
        s = season_state[team]
        points = 2 * s["wins"]
        played = s["wins"] + s["losses"]
        rs = s["runs_scored"]
        bf = s["balls_faced"]
        rc = s["runs_conceded"]
        bb = s["balls_bowled"]
        rr_for = (rs / (bf / 6)) if bf > 0 else 0.0
        rr_against = (rc / (bb / 6)) if bb > 0 else 0.0
        nrr = rr_for - rr_against
        standings.append((team, points, nrr, played))
    standings.sort(key=lambda x: (-x[1], -x[2]))
    return {team: rank + 1 for rank, (team, *_rest) in enumerate(standings)}


def build_features() -> pd.DataFrame:
    matches = pd.read_csv(PROC_DIR / "matches.csv", parse_dates=["date"])
    deliveries = pd.read_csv(PROC_DIR / "deliveries.csv")

    matches = matches[matches["winner"].notna()].copy()
    matches = matches.sort_values("date").reset_index(drop=True)

    scored = _team_run_rate(deliveries)
    pivot_runs = scored.pivot(index="match_id", columns="team", values="runs_scored")
    bf = _team_balls_faced(deliveries)
    pivot_balls = bf.pivot(index="match_id", columns="team", values="balls_faced")

    rows = []
    # rolling state
    last_n_results: dict[str, list[int]] = {}
    last_n_scored: dict[str, list[float]] = {}
    last_n_conceded: dict[str, list[float]] = {}
    venue_team_wins: dict[tuple, list[int]] = {}
    h2h: dict[tuple, list[int]] = {}
    toss_history: dict[str, list[int]] = {}

    # season state: season -> team -> {wins, losses, runs_scored, runs_conceded, balls_faced, balls_bowled}
    season_state: dict[str, dict[str, dict]] = {}

    def avg(lst: list[float], default=0.0, n: int = RECENT_N) -> float:
        return float(np.mean(lst[-n:])) if lst else default

    def empty_team_state() -> dict:
        return {"wins": 0, "losses": 0, "runs_scored": 0, "runs_conceded": 0,
                "balls_faced": 0, "balls_bowled": 0}

    def team_season_features(season: str, team: str) -> dict:
        state = season_state.get(season, {}).get(team)
        if not state:
            return {"wins": 0, "losses": 0, "points": 0, "winpct": 0.0,
                    "nrr": 0.0, "rank": 0, "played": 0}
        played = state["wins"] + state["losses"]
        rr_for = state["runs_scored"] / (state["balls_faced"] / 6) if state["balls_faced"] else 0.0
        rr_against = state["runs_conceded"] / (state["balls_bowled"] / 6) if state["balls_bowled"] else 0.0
        ranks = _season_table_snapshot(season_state.get(season, {}))
        return {
            "wins": state["wins"],
            "losses": state["losses"],
            "points": 2 * state["wins"],
            "winpct": state["wins"] / played if played else 0.0,
            "nrr": rr_for - rr_against,
            "rank": ranks.get(team, 0),
            "played": played,
        }

    for _, m in matches.iterrows():
        t1, t2, venue, winner = m["team1"], m["team2"], m["venue"], m["winner"]
        season = str(m["season"])
        mid = m["match_id"]

        s1 = team_season_features(season, t1)
        s2 = team_season_features(season, t2)

        f = {
            "match_id": mid,
            "season": season,
            "date": m["date"],
            "venue": venue,
            "team1": t1,
            "team2": t2,
            "t1_form": avg(last_n_results.get(t1, [])),
            "t2_form": avg(last_n_results.get(t2, [])),
            "t1_avg_scored": avg(last_n_scored.get(t1, []), 160.0),
            "t2_avg_scored": avg(last_n_scored.get(t2, []), 160.0),
            "t1_avg_conceded": avg(last_n_conceded.get(t1, []), 160.0),
            "t2_avg_conceded": avg(last_n_conceded.get(t2, []), 160.0),
            "t1_venue_winrate": avg(venue_team_wins.get((t1, venue), [])),
            "t2_venue_winrate": avg(venue_team_wins.get((t2, venue), [])),
            "h2h_t1_winrate": avg(h2h.get((t1, t2), [])),
            "t1_toss_winrate_10": avg(toss_history.get(t1, []), default=0.5, n=TOSS_N),
            "t2_toss_winrate_10": avg(toss_history.get(t2, []), default=0.5, n=TOSS_N),
            "toss_winner_is_t1": int(m["toss_winner"] == t1) if pd.notna(m["toss_winner"]) else 0,
            "toss_decision_bat": int(m["toss_decision"] == "bat") if pd.notna(m["toss_decision"]) else 0,
            # Points-table (kept in the file for reference/insights, not used by model)
            "t1_season_points": s1["points"],
            "t2_season_points": s2["points"],
            "t1_season_nrr": s1["nrr"],
            "t2_season_nrr": s2["nrr"],
            "t1_season_rank": s1["rank"],
            "t2_season_rank": s2["rank"],
            "target": 1 if winner == t1 else 0,
        }
        rows.append(f)

        # Update rolling state
        t1_won = 1 if winner == t1 else 0
        t2_won = 1 - t1_won
        last_n_results.setdefault(t1, []).append(t1_won)
        last_n_results.setdefault(t2, []).append(t2_won)

        t1_scored = t2_scored = np.nan
        t1_balls = t2_balls = np.nan
        if mid in pivot_runs.index:
            t1_scored = pivot_runs.loc[mid].get(t1, np.nan)
            t2_scored = pivot_runs.loc[mid].get(t2, np.nan)
        if mid in pivot_balls.index:
            t1_balls = pivot_balls.loc[mid].get(t1, np.nan)
            t2_balls = pivot_balls.loc[mid].get(t2, np.nan)

        if pd.notna(t1_scored):
            last_n_scored.setdefault(t1, []).append(float(t1_scored))
            last_n_conceded.setdefault(t2, []).append(float(t1_scored))
        if pd.notna(t2_scored):
            last_n_scored.setdefault(t2, []).append(float(t2_scored))
            last_n_conceded.setdefault(t1, []).append(float(t2_scored))

        venue_team_wins.setdefault((t1, venue), []).append(t1_won)
        venue_team_wins.setdefault((t2, venue), []).append(t2_won)
        h2h.setdefault((t1, t2), []).append(t1_won)
        h2h.setdefault((t2, t1), []).append(t2_won)

        if pd.notna(m["toss_winner"]):
            toss_history.setdefault(t1, []).append(int(m["toss_winner"] == t1))
            toss_history.setdefault(t2, []).append(int(m["toss_winner"] == t2))

        # Update season standings
        ss = season_state.setdefault(season, {})
        ss.setdefault(t1, empty_team_state())
        ss.setdefault(t2, empty_team_state())
        ss[t1]["wins"] += t1_won
        ss[t1]["losses"] += t2_won
        ss[t2]["wins"] += t2_won
        ss[t2]["losses"] += t1_won
        if pd.notna(t1_scored):
            ss[t1]["runs_scored"] += int(t1_scored)
            ss[t2]["runs_conceded"] += int(t1_scored)
        if pd.notna(t2_scored):
            ss[t2]["runs_scored"] += int(t2_scored)
            ss[t1]["runs_conceded"] += int(t2_scored)
        if pd.notna(t1_balls):
            ss[t1]["balls_faced"] += int(t1_balls)
            ss[t2]["balls_bowled"] += int(t1_balls)
        if pd.notna(t2_balls):
            ss[t2]["balls_faced"] += int(t2_balls)
            ss[t1]["balls_bowled"] += int(t2_balls)

    df = pd.DataFrame(rows)
    out = PROC_DIR / "features.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} feature rows to {out}")
    return df


if __name__ == "__main__":
    build_features()
