"""Full match insight pack: winner, score, top batter/bowler, 6s, 4s, pitch read.

All team / player / venue stats use IPL 2026 data only.
The winner classifier still trains on the full feature history (with proper
walk-forward setup) so it has enough samples to be reliable.

Usage:
    python -m src.match_insight --team1 "Gujarat Titans" \\
        --team2 "Royal Challengers Bengaluru" \\
        --venue "Narendra Modi Stadium, Ahmedabad" \\
        [--toss-winner "Gujarat Titans" --toss-decision field]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

NUMERIC = [
    "t1_form", "t2_form",
    "t1_avg_scored", "t2_avg_scored",
    "t1_avg_conceded", "t2_avg_conceded",
    "t1_venue_winrate", "t2_venue_winrate",
    "h2h_t1_winrate",
    "t1_toss_winrate_10", "t2_toss_winrate_10",
    "toss_winner_is_t1", "toss_decision_bat",
]
CATS = ["team1", "team2", "venue"]
SEASON = "2026"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def load_2026(as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    deliveries = pd.read_csv(PROC / "deliveries.csv")
    m26 = matches[matches["season"].astype(str) == SEASON].copy()
    if as_of is not None:
        m26 = m26[m26["date"] < as_of]
    d26 = deliveries[deliveries["match_id"].isin(m26["match_id"])].copy()
    return m26, d26


def team_recent_form(m26: pd.DataFrame, team: str, n: int = 5) -> dict:
    games = m26[(m26["team1"] == team) | (m26["team2"] == team)].sort_values("date").tail(n)
    if games.empty:
        return {"played": 0, "wins": 0, "form": "—", "results": []}
    results = []
    wins = 0
    for _, g in games.iterrows():
        opp = g["team2"] if g["team1"] == team else g["team1"]
        won = g["winner"] == team
        wins += int(won)
        results.append(f"{'W' if won else 'L'} vs {opp[:3]}")
    return {"played": len(games), "wins": wins, "form": f"{wins}-{len(games) - wins}",
            "results": results}


def team_scoring(d26: pd.DataFrame, m26: pd.DataFrame, team: str) -> dict:
    # Innings batted by team
    batted = d26[d26["batting_team"] == team]
    if batted.empty:
        return {"avg_scored": 0, "avg_conceded": 0, "avg_6s": 0, "avg_4s": 0, "innings": 0}
    inn = batted.groupby("match_id").agg(
        runs=("runs_total", "sum"),
        sixes=("runs_batter", lambda s: (s == 6).sum()),
        fours=("runs_batter", lambda s: (s == 4).sum()),
    )
    # Conceded = opponents batting in same matches
    team_matches = m26[(m26["team1"] == team) | (m26["team2"] == team)]["match_id"]
    conceded = d26[(d26["match_id"].isin(team_matches)) & (d26["batting_team"] != team)]
    conc_inn = conceded.groupby("match_id")["runs_total"].sum()
    return {
        "avg_scored": float(inn["runs"].mean()),
        "avg_conceded": float(conc_inn.mean()) if not conc_inn.empty else 0.0,
        "avg_6s": float(inn["sixes"].mean()),
        "avg_4s": float(inn["fours"].mean()),
        "innings": int(len(inn)),
    }


def venue_profile(d26: pd.DataFrame, m26: pd.DataFrame, venue: str) -> dict:
    venue_m = m26[m26["venue"] == venue]
    if venue_m.empty:
        return None
    venue_d = d26[d26["match_id"].isin(venue_m["match_id"])]
    inns = venue_d.groupby(["match_id", "inning"])["runs_total"].sum().reset_index()
    first_inn = inns[inns["inning"] == 1]["runs_total"]
    sixes_per_match = venue_d.groupby("match_id")["runs_batter"].apply(lambda s: (s == 6).sum())
    fours_per_match = venue_d.groupby("match_id")["runs_batter"].apply(lambda s: (s == 4).sum())
    wkts_per_match = venue_d.groupby("match_id")["wicket"].sum()
    runs_per_match = venue_d.groupby("match_id")["runs_total"].sum()
    return {
        "matches": len(venue_m),
        "avg_first_inn": float(first_inn.mean()) if not first_inn.empty else 0,
        "avg_total_runs": float(runs_per_match.mean()),
        "avg_sixes": float(sixes_per_match.mean()),
        "avg_fours": float(fours_per_match.mean()),
        "avg_wickets": float(wkts_per_match.mean()),
        "runs_per_wicket": float(runs_per_match.mean() / max(wkts_per_match.mean(), 1)),
    }


def top_batters(d26: pd.DataFrame, team: str, n: int = 3) -> pd.DataFrame:
    bat = d26[d26["batting_team"] == team]
    if bat.empty:
        return pd.DataFrame()
    g = bat.groupby("batter").agg(
        runs=("runs_batter", "sum"),
        balls=("runs_batter", "count"),
        sixes=("runs_batter", lambda s: (s == 6).sum()),
        fours=("runs_batter", lambda s: (s == 4).sum()),
        innings=("match_id", "nunique"),
    )
    g["sr"] = (g["runs"] / g["balls"] * 100).round(1)
    g["avg"] = (g["runs"] / g["innings"]).round(1)
    g = g[g["innings"] >= 3].sort_values("runs", ascending=False)
    return g.head(n)


def top_bowlers(d26: pd.DataFrame, team: str, m26: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    # bowler is on the OPPOSITE team from batting_team
    team_matches = m26[(m26["team1"] == team) | (m26["team2"] == team)]["match_id"]
    bowled = d26[(d26["match_id"].isin(team_matches)) & (d26["batting_team"] != team)]
    if bowled.empty:
        return pd.DataFrame()
    g = bowled.groupby("bowler").agg(
        runs=("runs_total", "sum"),
        balls=("runs_total", "count"),
        wickets=("wicket", "sum"),
        innings=("match_id", "nunique"),
    )
    g["overs"] = (g["balls"] / 6).round(1)
    g["er"] = (g["runs"] / (g["balls"] / 6)).round(2)
    g = g[g["innings"] >= 3].sort_values(["wickets", "er"], ascending=[False, True])
    return g.head(n)


def predict_winner(team1: str, team2: str, venue: str,
                   toss_winner: str | None, toss_decision: str | None,
                   as_of: pd.Timestamp | None = None) -> tuple[float, dict]:
    df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    if as_of is not None:
        df = df[df["date"] < as_of]

    encoders = {}
    for c in CATS:
        le = LabelEncoder()
        le.fit(df[c].astype(str))
        df[c + "_enc"] = le.transform(df[c].astype(str))
        encoders[c] = le

    # ENSEMBLE: average two models — full features (with team identity)
    # and an "anti-brand" version that drops team_enc / team2_enc.
    full_cols = NUMERIC + [c + "_enc" for c in CATS]
    no_brand_cols = NUMERIC + ["venue_enc"]
    model_full = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
    model_full.fit(df[full_cols], df["target"])
    model_nb = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
    model_nb.fit(df[no_brand_cols], df["target"])
    feat_cols = full_cols  # for the row-build below

    # Build feature row for today's match using latest 2026-rolling state.
    # Easiest: take the most recent feature row that includes team1 (as either side)
    # and project stats. We'll replicate the rolling logic just for these two teams.

    # Use the latest entries to read state
    def latest(team: str, opp: str) -> dict:
        # find the most recent row featuring `team` (as t1 or t2)
        recent = df[(df["team1"] == team) | (df["team2"] == team)].sort_values("date").iloc[-1]
        if recent["team1"] == team:
            return {
                "form": recent["t1_form"],
                "scored": recent["t1_avg_scored"],
                "conceded": recent["t1_avg_conceded"],
                "toss_rate": recent["t1_toss_winrate_10"],
            }
        return {
            "form": recent["t2_form"],
            "scored": recent["t2_avg_scored"],
            "conceded": recent["t2_avg_conceded"],
            "toss_rate": recent["t2_toss_winrate_10"],
        }

    s1 = latest(team1, team2)
    s2 = latest(team2, team1)

    # Venue winrate from 2026 matches at this venue
    m26, _ = load_2026(as_of)
    vm = m26[m26["venue"] == venue]
    t1_vw = (vm["winner"] == team1).mean() if not vm.empty else 0.5
    t2_vw = (vm["winner"] == team2).mean() if not vm.empty else 0.5

    # H2H from 2026 only
    h2h_games = m26[((m26["team1"] == team1) & (m26["team2"] == team2)) |
                    ((m26["team1"] == team2) & (m26["team2"] == team1))]
    if not h2h_games.empty:
        h2h_rate = (h2h_games["winner"] == team1).mean()
    else:
        h2h_rate = 0.5

    row = {
        "t1_form": s1["form"], "t2_form": s2["form"],
        "t1_avg_scored": s1["scored"], "t2_avg_scored": s2["scored"],
        "t1_avg_conceded": s1["conceded"], "t2_avg_conceded": s2["conceded"],
        "t1_venue_winrate": t1_vw, "t2_venue_winrate": t2_vw,
        "h2h_t1_winrate": h2h_rate,
        "t1_toss_winrate_10": s1["toss_rate"], "t2_toss_winrate_10": s2["toss_rate"],
        "toss_winner_is_t1": int(toss_winner == team1) if toss_winner else 0,
        "toss_decision_bat": int(toss_decision == "bat") if toss_decision else 0,
    }
    for c, v in [("team1", team1), ("team2", team2), ("venue", venue)]:
        le = encoders[c]
        row[c + "_enc"] = int(le.transform([v])[0]) if v in le.classes_ else -1

    X_full = pd.DataFrame([row])[full_cols]
    X_nb = pd.DataFrame([row])[no_brand_cols]
    p_full = float(model_full.predict_proba(X_full)[0, 1])
    p_nb = float(model_nb.predict_proba(X_nb)[0, 1])
    p1 = (p_full + p_nb) / 2
    return p1, {"h2h_2026_t1_winrate": h2h_rate, "venue_t1_winrate_2026": t1_vw,
                "venue_t2_winrate_2026": t2_vw,
                "p_full": p_full, "p_no_brand": p_nb}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team1", required=True)
    p.add_argument("--team2", required=True)
    p.add_argument("--venue", required=True)
    p.add_argument("--toss-winner")
    p.add_argument("--toss-decision", choices=["bat", "field"])
    p.add_argument("--as-of", help="YYYY-MM-DD; only use data BEFORE this date (for back-test)")
    args = p.parse_args()

    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    m26, d26 = load_2026(as_of)
    t1, t2, venue = args.team1, args.team2, args.venue

    section(f"MATCH PREVIEW: {t1} vs {t2}")
    print(f"Venue: {venue}")
    print(f"Data window: IPL {SEASON} only ({len(m26)} matches, {len(d26)} deliveries)")

    # Form
    section("RECENT FORM (last 5 in IPL 2026)")
    for team in (t1, t2):
        f = team_recent_form(m26, team)
        print(f"{team:30s}  {f['form']:>5s}  | {' '.join(f['results'])}")

    # Team scoring
    section("TEAM SCORING (IPL 2026)")
    t1s = team_scoring(d26, m26, t1)
    t2s = team_scoring(d26, m26, t2)
    print(f"{'':30s}  Avg scored  Avg conceded  Avg 6s  Avg 4s  Innings")
    print(f"{t1:30s}  {t1s['avg_scored']:>10.1f}  {t1s['avg_conceded']:>12.1f}  "
          f"{t1s['avg_6s']:>6.1f}  {t1s['avg_4s']:>6.1f}  {t1s['innings']:>7d}")
    print(f"{t2:30s}  {t2s['avg_scored']:>10.1f}  {t2s['avg_conceded']:>12.1f}  "
          f"{t2s['avg_6s']:>6.1f}  {t2s['avg_4s']:>6.1f}  {t2s['innings']:>7d}")

    # Venue
    section("PITCH / VENUE PROFILE (IPL 2026)")
    vp = venue_profile(d26, m26, venue)
    if vp:
        print(f"Matches at venue : {vp['matches']}")
        print(f"Avg 1st innings  : {vp['avg_first_inn']:.0f}")
        print(f"Avg total runs   : {vp['avg_total_runs']:.0f}")
        print(f"Avg sixes/match  : {vp['avg_sixes']:.1f}")
        print(f"Avg fours/match  : {vp['avg_fours']:.1f}")
        print(f"Avg wickets/match: {vp['avg_wickets']:.1f}")
        print(f"Runs/wicket      : {vp['runs_per_wicket']:.1f}  "
              f"({'batting-friendly' if vp['runs_per_wicket'] > 28 else 'bowler-friendly' if vp['runs_per_wicket'] < 22 else 'balanced'})")
    else:
        print(f"No 2026 matches recorded at {venue}.")

    # Top batters / bowlers
    section("TOP BATTERS (IPL 2026)")
    for team in (t1, t2):
        print(f"\n{team}:")
        tb = top_batters(d26, team)
        if tb.empty:
            print("  (insufficient data)")
        else:
            print(tb[["runs", "innings", "avg", "sr", "sixes", "fours"]].to_string())

    section("TOP BOWLERS (IPL 2026)")
    for team in (t1, t2):
        print(f"\n{team}:")
        tb = top_bowlers(d26, team, m26)
        if tb.empty:
            print("  (insufficient data)")
        else:
            print(tb[["overs", "wickets", "er", "innings"]].to_string())

    # Winner prediction
    section("WINNER PREDICTION")
    p1, ctx = predict_winner(t1, t2, venue, args.toss_winner, args.toss_decision, as_of)
    print(f"P({t1}) = {p1:.2f}")
    print(f"P({t2}) = {1 - p1:.2f}")
    print(f">> Predicted winner: {t1 if p1 >= 0.5 else t2}")

    # Score / 6s / 4s projection (blend team avg + venue avg)
    section("SCORE & BOUNDARY PROJECTION")
    avg_total = vp["avg_first_inn"] if vp else 165
    t1_proj = 0.5 * t1s["avg_scored"] + 0.5 * avg_total
    t2_proj = 0.5 * t2s["avg_scored"] + 0.5 * avg_total
    # Each team bats once -> sum per-innings averages = expected match total
    teams_sixes = t1s["avg_6s"] + t2s["avg_6s"]
    teams_fours = t1s["avg_4s"] + t2s["avg_4s"]
    venue_sixes = vp["avg_sixes"] if vp else 14
    venue_fours = vp["avg_fours"] if vp else 26
    six_proj = (teams_sixes + venue_sixes) / 2
    four_proj = (teams_fours + venue_fours) / 2
    print(f"Predicted 1st-innings total : {t1_proj:.0f}-{t2_proj:.0f} (range {min(t1_proj, t2_proj) - 15:.0f}-{max(t1_proj, t2_proj) + 15:.0f})")
    print(f"Predicted total sixes       : {six_proj:.0f}")
    print(f"Predicted total fours       : {four_proj:.0f}")

    print()
    print("Caveats: predictions assume both teams field full-strength sides; injuries / "
          "playing-XI changes can shift the outcome. Score and boundary numbers are point "
          "estimates, expect ±20 runs / ±5 boundaries variance.")


if __name__ == "__main__":
    main()
