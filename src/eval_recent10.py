"""Walk-forward prediction of the LAST 10 completed IPL 2026 matches
using the ENSEMBLE model (Baseline + Drop-team-identity, averaged).

For each test match, both component models retrain on every match that happened
strictly BEFORE it (2023, 2024, 2025, and earlier 2026 matches). Their
probabilities are averaged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"

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


def main() -> None:
    df = pd.read_csv(PROC_DIR / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)

    matches = pd.read_csv(PROC_DIR / "matches.csv", parse_dates=["date"])

    encoders = {}
    for c in CATS:
        le = LabelEncoder()
        le.fit(df[c].astype(str))
        df[c + "_enc"] = le.transform(df[c].astype(str))
        encoders[c] = le

    # All completed 2026 matches (winner is non-null), pick LAST 10 by date
    completed_2026 = df[df["season"].astype(str) == "2026"].sort_values("date")
    test = completed_2026.tail(10).reset_index(drop=True)
    print(f"Last 10 completed IPL 2026 matches (from {test['date'].min().date()} "
          f"to {test['date'].max().date()})\n")

    feat_cols = NUMERIC + [c + "_enc" for c in CATS]

    no_brand_cols = [c for c in feat_cols if c not in ("team1_enc", "team2_enc")]

    rows = []
    correct = 0
    for i, r in test.iterrows():
        cutoff_date = r["date"]
        train = df[df["date"] < cutoff_date]

        m_full = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
        m_full.fit(train[feat_cols], train["target"])
        m_nb = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
        m_nb.fit(train[no_brand_cols], train["target"])

        X_full = pd.DataFrame([r[feat_cols].values], columns=feat_cols)
        X_nb = pd.DataFrame([r[no_brand_cols].values], columns=no_brand_cols)
        p_full = float(m_full.predict_proba(X_full)[0, 1])
        p_nb = float(m_nb.predict_proba(X_nb)[0, 1])
        p1 = (p_full + p_nb) / 2  # ensemble (50/50)
        predicted = r["team1"] if p1 >= 0.5 else r["team2"]

        actual = matches.loc[matches["match_id"] == r["match_id"], "winner"].iloc[0]
        ok = predicted == actual
        if ok:
            correct += 1

        toss_w = matches.loc[matches["match_id"] == r["match_id"], "toss_winner"].iloc[0]
        toss_d = matches.loc[matches["match_id"] == r["match_id"], "toss_decision"].iloc[0]

        rows.append({
            "#": i + 1,
            "date": r["date"].date(),
            "match": f"{r['team1']} vs {r['team2']}",
            "toss": f"{toss_w[:3] if isinstance(toss_w, str) else '?'} ({toss_d})",
            "t1_pts/rnk": f"{int(r['t1_season_points'])}/{int(r['t1_season_rank'])}",
            "t2_pts/rnk": f"{int(r['t2_season_points'])}/{int(r['t2_season_rank'])}",
            "t1_NRR": round(r["t1_season_nrr"], 2),
            "t2_NRR": round(r["t2_season_nrr"], 2),
            "t1_form": round(r["t1_form"], 2),
            "t2_form": round(r["t2_form"], 2),
            "P(t1)": round(p1, 2),
            "predicted": predicted,
            "actual": actual,
            "✓": "✓" if ok else "✗",
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"\nAccuracy on last 10 completed IPL 2026 matches: "
          f"{correct}/{len(test)} = {correct / len(test):.1%}")


if __name__ == "__main__":
    main()
