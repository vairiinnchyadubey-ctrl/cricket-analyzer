"""Walk-forward prediction of the LAST 20 completed IPL 2026 matches.

For each test match, the model is retrained on every match that happened
strictly BEFORE it. Features used: form, H2H, venue, scoring/conceding rates,
toss winner, toss decision, and rolling toss-win rate (last 10).
"""
from __future__ import annotations

from pathlib import Path

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
N_TEST = 20


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

    feat_cols = NUMERIC + [c + "_enc" for c in CATS]
    completed_2026 = df[df["season"].astype(str) == "2026"].sort_values("date")
    test = completed_2026.tail(N_TEST).reset_index(drop=True)

    print(f"Last {N_TEST} completed IPL 2026 matches "
          f"({test['date'].min().date()} to {test['date'].max().date()})\n")

    rows = []
    correct = 0
    for i, r in test.iterrows():
        train = df[df["date"] < r["date"]]
        model = GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42,
        )
        model.fit(train[feat_cols], train["target"])
        X = pd.DataFrame([r[feat_cols].values], columns=feat_cols)
        p1 = float(model.predict_proba(X)[0, 1])
        predicted = r["team1"] if p1 >= 0.5 else r["team2"]
        actual = matches.loc[matches["match_id"] == r["match_id"], "winner"].iloc[0]
        toss_w = matches.loc[matches["match_id"] == r["match_id"], "toss_winner"].iloc[0]
        toss_d = matches.loc[matches["match_id"] == r["match_id"], "toss_decision"].iloc[0]
        ok = predicted == actual
        if ok:
            correct += 1
        rows.append({
            "#": i + 1,
            "date": r["date"].date(),
            "match": f"{r['team1']} vs {r['team2']}",
            "toss": f"{toss_w[:3] if isinstance(toss_w, str) else '?'} ({toss_d})",
            "t1_form": round(r["t1_form"], 2),
            "t2_form": round(r["t2_form"], 2),
            "h2h": round(r["h2h_t1_winrate"], 2),
            "P(t1)": round(p1, 2),
            "predicted": predicted,
            "actual": actual,
            "✓": "✓" if ok else "✗",
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"\nAccuracy on last {N_TEST} completed IPL 2026 matches: "
          f"{correct}/{len(test)} = {correct / len(test):.1%}")


if __name__ == "__main__":
    main()
