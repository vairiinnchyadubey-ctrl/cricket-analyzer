"""Kitchen-sink feature builder: every reasonable signal we can compute.

For each match we track per-team rolling state (last 5 unless noted):
  Existing (in features.csv):
    form, avg_scored, avg_conceded, venue_winrate, h2h_winrate,
    toss_winrate_10, toss_winner_is_t1, toss_decision_bat

  Phase-wise scoring (NEW):
    pp_rr_for / pp_rr_against     (Powerplay run rates)
    middle_rr_for / against
    death_rr_for / against

  Form-color (NEW):
    win_streak                    (consecutive wins as of now)
    avg_margin                    (avg win margin in runs/wkts last 5; +ve = dominant)
    boundary_pct_for / against    (4s+6s as % of balls last 5)
    dot_pct_for / against
    wickets_taken_last5
    wickets_lost_last5

  Match context (NEW):
    days_since_last              (fewer = fatigued)
    matches_played_in_season     (tournament progression)
    is_home_venue                (team's home ground? heuristic)

Output: data/processed/ks_features.csv  (one row per match_id, with all numeric cols)
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

WIN = 5

# Crude home-venue mapping
HOME_VENUE = {
    "Mumbai Indians":               "Wankhede Stadium, Mumbai",
    "Chennai Super Kings":          "MA Chidambaram Stadium, Chepauk, Chennai",
    "Royal Challengers Bengaluru":  "M Chinnaswamy Stadium, Bengaluru",
    "Royal Challengers Bangalore":  "M Chinnaswamy Stadium, Bengaluru",
    "Kolkata Knight Riders":        "Eden Gardens, Kolkata",
    "Sunrisers Hyderabad":          "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "Delhi Capitals":               "Arun Jaitley Stadium, Delhi",
    "Punjab Kings":                 "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh",
    "Rajasthan Royals":             "Sawai Mansingh Stadium, Jaipur",
    "Lucknow Super Giants":         "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Gujarat Titans":               "Narendra Modi Stadium, Ahmedabad",
}


def _agg_balls(deliveries: pd.DataFrame, mid: str, team: str, mode: str = "for"):
    """Per-match balls / runs for a team (mode='for' batting, 'against' bowling)."""
    if mode == "for":
        sub = deliveries[(deliveries["match_id"] == mid) & (deliveries["batting_team"] == team)]
    else:
        sub = deliveries[(deliveries["match_id"] == mid) & (deliveries["batting_team"] != team)]
    return sub


def main() -> None:
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    matches = matches[matches["winner"].notna()].sort_values("date").reset_index(drop=True)
    deliveries = pd.read_csv(PROC / "deliveries.csv")

    # Index deliveries by (match_id, batting_team) for speed
    pp_mask = deliveries["over"] < 6
    mid_mask = (deliveries["over"] >= 6) & (deliveries["over"] < 15)
    de_mask = deliveries["over"] >= 15

    rows = []
    last_n_results: dict[str, list[int]] = defaultdict(list)
    last_n_margins: dict[str, list[float]] = defaultdict(list)
    last_n_pp_rf:   dict[str, list[float]] = defaultdict(list)
    last_n_pp_ra:   dict[str, list[float]] = defaultdict(list)
    last_n_md_rf:   dict[str, list[float]] = defaultdict(list)
    last_n_md_ra:   dict[str, list[float]] = defaultdict(list)
    last_n_de_rf:   dict[str, list[float]] = defaultdict(list)
    last_n_de_ra:   dict[str, list[float]] = defaultdict(list)
    last_n_b_for:   dict[str, list[float]] = defaultdict(list)
    last_n_b_aga:   dict[str, list[float]] = defaultdict(list)
    last_n_d_for:   dict[str, list[float]] = defaultdict(list)
    last_n_d_aga:   dict[str, list[float]] = defaultdict(list)
    last_n_wt:      dict[str, list[int]]   = defaultdict(list)
    last_n_wl:      dict[str, list[int]]   = defaultdict(list)
    last_match_date:dict[str, pd.Timestamp]= {}
    matches_in_season:dict[tuple, int]     = defaultdict(int)

    def avg(lst, default=0.0):
        return float(np.mean(lst[-WIN:])) if lst else default

    def streak(lst):
        s = 0
        for r in reversed(lst):
            if r == 1: s += 1
            else: break
        return s

    for _, m in tqdm(matches.iterrows(), total=len(matches), desc="kitchen-sink"):
        t1, t2, mid, d, season = m["team1"], m["team2"], m["match_id"], m["date"], str(m["season"])

        # Pull pre-match values
        def features_for(team, opp):
            return {
                "form": avg(last_n_results[team]),
                "win_streak": streak(last_n_results[team]),
                "avg_margin": avg(last_n_margins[team]),
                "pp_rr_for":     avg(last_n_pp_rf[team], 8.5),
                "pp_rr_against": avg(last_n_pp_ra[team], 8.5),
                "md_rr_for":     avg(last_n_md_rf[team], 8.0),
                "md_rr_against": avg(last_n_md_ra[team], 8.0),
                "de_rr_for":     avg(last_n_de_rf[team], 9.5),
                "de_rr_against": avg(last_n_de_ra[team], 9.5),
                "boundary_pct_for":     avg(last_n_b_for[team], 18.0),
                "boundary_pct_against": avg(last_n_b_aga[team], 18.0),
                "dot_pct_for":     avg(last_n_d_for[team], 35.0),
                "dot_pct_against": avg(last_n_d_aga[team], 35.0),
                "wickets_taken_5":  avg(last_n_wt[team], 6.0),
                "wickets_lost_5":   avg(last_n_wl[team], 6.0),
                "days_since_last": (d - last_match_date[team]).days if team in last_match_date else 30,
                "season_match_no": matches_in_season[(team, season)],
                "is_home_venue":   int(HOME_VENUE.get(team) == m["venue"]),
            }

        f1 = features_for(t1, t2)
        f2 = features_for(t2, t1)
        row = {"match_id": mid}
        for k, v in f1.items(): row[f"t1_{k}"] = v
        for k, v in f2.items(): row[f"t2_{k}"] = v
        # diff features
        row["form_diff"]     = f1["form"]      - f2["form"]
        row["streak_diff"]   = f1["win_streak"] - f2["win_streak"]
        row["pp_for_diff"]   = f1["pp_rr_for"] - f2["pp_rr_for"]
        row["death_for_diff"]= f1["de_rr_for"] - f2["de_rr_for"]
        row["margin_diff"]   = f1["avg_margin"]- f2["avg_margin"]
        rows.append(row)

        # Update state with this match's outcomes
        winner = m["winner"]; t1_won = int(winner == t1)
        # Margin proxy: positive if team won, magnitude = run-margin or 5*wkt-margin
        margin_runs = m.get("win_by_runs"); margin_wkts = m.get("win_by_wickets")
        margin = float(margin_runs) if pd.notna(margin_runs) else (5.0 * float(margin_wkts) if pd.notna(margin_wkts) else 5.0)

        for team, won in [(t1, t1_won), (t2, 1 - t1_won)]:
            last_n_results[team].append(won)
            last_n_margins[team].append(margin if won else -margin)

            # Phase rates: scored
            for_sub = _agg_balls(deliveries, mid, team, "for")
            against_sub = _agg_balls(deliveries, mid, team, "against")

            for_pp = for_sub[for_sub["over"] < 6]
            for_md = for_sub[(for_sub["over"] >= 6) & (for_sub["over"] < 15)]
            for_de = for_sub[for_sub["over"] >= 15]

            ag_pp = against_sub[against_sub["over"] < 6]
            ag_md = against_sub[(against_sub["over"] >= 6) & (against_sub["over"] < 15)]
            ag_de = against_sub[against_sub["over"] >= 15]

            def rr(sub):
                return (sub["runs_total"].sum() / max(len(sub),1)) * 6 if len(sub) else 8.0

            last_n_pp_rf[team].append(rr(for_pp))
            last_n_pp_ra[team].append(rr(ag_pp))
            last_n_md_rf[team].append(rr(for_md))
            last_n_md_ra[team].append(rr(ag_md))
            last_n_de_rf[team].append(rr(for_de))
            last_n_de_ra[team].append(rr(ag_de))

            # Boundary / dot %
            if len(for_sub):
                b = ((for_sub["runs_batter"] == 4) | (for_sub["runs_batter"] == 6)).sum() / len(for_sub) * 100
                dt = (for_sub["runs_total"] == 0).sum() / len(for_sub) * 100
                last_n_b_for[team].append(float(b))
                last_n_d_for[team].append(float(dt))
            if len(against_sub):
                b = ((against_sub["runs_batter"] == 4) | (against_sub["runs_batter"] == 6)).sum() / len(against_sub) * 100
                dt = (against_sub["runs_total"] == 0).sum() / len(against_sub) * 100
                last_n_b_aga[team].append(float(b))
                last_n_d_aga[team].append(float(dt))

            # Wickets
            wt = int(against_sub["wicket"].sum()) if len(against_sub) else 0
            wl = int(for_sub["wicket"].sum()) if len(for_sub) else 0
            last_n_wt[team].append(wt)
            last_n_wl[team].append(wl)

            last_match_date[team] = d
            matches_in_season[(team, season)] += 1

    df = pd.DataFrame(rows)
    out = PROC / "ks_features.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} kitchen-sink feature rows ({len(df.columns)-1} numeric cols) → {out}")


if __name__ == "__main__":
    main()
