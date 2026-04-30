"""Train on IPL 2023-2025 and blind-predict the first 10 matches of IPL 2026.

Compares predicted winner vs actual winner and reports accuracy.
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

    encoders = {}
    for c in CATS:
        le = LabelEncoder()
        df[c + "_enc"] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    feat_cols = NUMERIC + [c + "_enc" for c in CATS]

    train = df[df["season"].astype(str).isin(["2023", "2024", "2025"])]
    test = df[df["season"].astype(str) == "2026"].sort_values("date").head(10)

    print(f"Train rows: {len(train)}   Test rows (first 10 of 2026): {len(test)}")

    model = GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42,
    )
    model.fit(train[feat_cols], train["target"])

    proba = model.predict_proba(test[feat_cols])[:, 1]
    pred = (proba >= 0.5).astype(int)

    matches = pd.read_csv(PROC_DIR / "matches.csv", parse_dates=["date"])
    test = test.merge(matches[["match_id", "winner"]], on="match_id", how="left")

    out = []
    correct = 0
    for i, (_, r) in enumerate(test.iterrows(), start=1):
        p1 = float(proba[i - 1])
        predicted = r["team1"] if pred[i - 1] == 1 else r["team2"]
        actual = r["winner"]
        ok = "✓" if predicted == actual else "✗"
        if predicted == actual:
            correct += 1
        out.append({
            "#": i,
            "date": r["date"].date(),
            "match": f"{r['team1']} vs {r['team2']}",
            "venue": r["venue"][:40],
            "P(team1)": round(p1, 2),
            "predicted": predicted,
            "actual": actual,
            "result": ok,
        })

    print()
    print(pd.DataFrame(out).to_string(index=False))
    print()
    print(f"Accuracy on first 10 matches of IPL 2026: {correct}/{len(test)} = {correct / max(len(test),1):.1%}")


if __name__ == "__main__":
    main()
