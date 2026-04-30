"""Predict the winner of an upcoming IPL match.

Usage:
    python -m src.predict --team1 "Mumbai Indians" --team2 "Chennai Super Kings" \\
        --venue "Wankhede Stadium, Mumbai" --toss-winner "Mumbai Indians" --toss-decision bat
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

RECENT_N = 5


def latest_team_state(features_df: pd.DataFrame, team: str, venue: str) -> dict:
    """Pull the most recent rolling stats for a team from the feature table."""
    rows_t1 = features_df[features_df["team1"] == team].sort_values("date").tail(RECENT_N)
    rows_t2 = features_df[features_df["team2"] == team].sort_values("date").tail(RECENT_N)

    form, scored, conceded = [], [], []
    for _, r in rows_t1.iterrows():
        form.append(r["target"])
        scored.append(r["t1_avg_scored"])
        conceded.append(r["t1_avg_conceded"])
    for _, r in rows_t2.iterrows():
        form.append(1 - r["target"])
        scored.append(r["t2_avg_scored"])
        conceded.append(r["t2_avg_conceded"])

    venue_rows_1 = features_df[(features_df["team1"] == team) & (features_df["venue"] == venue)]
    venue_rows_2 = features_df[(features_df["team2"] == team) & (features_df["venue"] == venue)]
    venue_wins = list(venue_rows_1["target"]) + list(1 - venue_rows_2["target"])

    return {
        "form": float(np.mean(form)) if form else 0.5,
        "scored": float(np.mean(scored)) if scored else 160.0,
        "conceded": float(np.mean(conceded)) if conceded else 160.0,
        "venue_winrate": float(np.mean(venue_wins[-RECENT_N:])) if venue_wins else 0.5,
    }


def h2h_rate(features_df: pd.DataFrame, t1: str, t2: str) -> float:
    a = features_df[(features_df["team1"] == t1) & (features_df["team2"] == t2)]["target"]
    b = 1 - features_df[(features_df["team1"] == t2) & (features_df["team2"] == t1)]["target"]
    combined = list(a) + list(b)
    return float(np.mean(combined[-RECENT_N:])) if combined else 0.5


def predict(team1: str, team2: str, venue: str,
            toss_winner: str | None, toss_decision: str | None) -> dict:
    bundle = joblib.load(MODEL_DIR / "winner_model.joblib")
    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_cols = bundle["feature_cols"]

    features_df = pd.read_csv(PROC_DIR / "features.csv", parse_dates=["date"])

    s1 = latest_team_state(features_df, team1, venue)
    s2 = latest_team_state(features_df, team2, venue)
    h2h = h2h_rate(features_df, team1, team2)

    row = {
        "t1_form": s1["form"], "t2_form": s2["form"],
        "t1_avg_scored": s1["scored"], "t2_avg_scored": s2["scored"],
        "t1_avg_conceded": s1["conceded"], "t2_avg_conceded": s2["conceded"],
        "t1_venue_winrate": s1["venue_winrate"], "t2_venue_winrate": s2["venue_winrate"],
        "h2h_t1_winrate": h2h,
        "toss_winner_is_t1": int(toss_winner == team1) if toss_winner else 0,
        "toss_decision_bat": int(toss_decision == "bat") if toss_decision else 0,
    }
    for col, val in [("team1", team1), ("team2", team2), ("venue", venue)]:
        le = encoders[col]
        if val in le.classes_:
            row[col + "_enc"] = int(le.transform([val])[0])
        else:
            row[col + "_enc"] = -1  # unseen
            print(f"[warn] unseen {col}: {val}")

    X = pd.DataFrame([row])[feature_cols]
    p1 = float(model.predict_proba(X)[0, 1])
    winner = team1 if p1 >= 0.5 else team2

    return {
        "team1": team1, "team2": team2, "venue": venue,
        "p_team1_wins": round(p1, 3),
        "p_team2_wins": round(1 - p1, 3),
        "predicted_winner": winner,
        "insights": {
            f"{team1} recent form (win %)": round(s1["form"] * 100, 1),
            f"{team2} recent form (win %)": round(s2["form"] * 100, 1),
            f"{team1} avg scored / conceded": (round(s1["scored"], 1), round(s1["conceded"], 1)),
            f"{team2} avg scored / conceded": (round(s2["scored"], 1), round(s2["conceded"], 1)),
            f"{team1} win % at {venue}": round(s1["venue_winrate"] * 100, 1),
            f"{team2} win % at {venue}": round(s2["venue_winrate"] * 100, 1),
            f"H2H ({team1} vs {team2})": f"{round(h2h * 100, 1)}% to {team1}",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team1", required=True)
    p.add_argument("--team2", required=True)
    p.add_argument("--venue", required=True)
    p.add_argument("--toss-winner")
    p.add_argument("--toss-decision", choices=["bat", "field"])
    args = p.parse_args()

    out = predict(args.team1, args.team2, args.venue, args.toss_winner, args.toss_decision)
    print("\n=== Prediction ===")
    print(f"{out['team1']} vs {out['team2']} @ {out['venue']}")
    print(f"P({out['team1']}) = {out['p_team1_wins']}")
    print(f"P({out['team2']}) = {out['p_team2_wins']}")
    print(f">> Predicted winner: {out['predicted_winner']}")
    print("\n=== Insights ===")
    for k, v in out["insights"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
