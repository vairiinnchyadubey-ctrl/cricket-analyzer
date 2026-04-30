"""Team-level only comparison of variants for winner prediction.

  A · Baseline                     all features, all years, uniform weights
  B · Light recency                weights 2026=2, 2025=1.5, 2024=1, 2023=0.7
  C · Drop team identity           remove team1_enc/team2_enc (kill brand bias)
  D · Recent years only            drop IPL 2023, train on 2024-2026
  E · C + D combined               no team identity + no 2023
  F · Ensemble (A + C)             average their probabilities
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

NUMERIC = [
    "t1_form","t2_form","t1_avg_scored","t2_avg_scored",
    "t1_avg_conceded","t2_avg_conceded","t1_venue_winrate","t2_venue_winrate",
    "h2h_t1_winrate","t1_toss_winrate_10","t2_toss_winrate_10",
    "toss_winner_is_t1","toss_decision_bat",
]
CATS = ["team1","team2","venue"]
LIGHT_WEIGHTS = {"2023": 0.7, "2024": 1.0, "2025": 1.5, "2026": 2.0}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    for c in CATS:
        le = LabelEncoder(); le.fit(df[c].astype(str))
        df[c+"_enc"] = le.transform(df[c].astype(str))
    return df, matches


def fit_predict(train, test_row, feat_cols, weights=None):
    m = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
    m.fit(train[feat_cols], train["target"], sample_weight=weights)
    X = pd.DataFrame([test_row[feat_cols].values], columns=feat_cols)
    return float(m.predict_proba(X)[0,1])


def variant(df, matches, test, *, drop_team_id=False, light_recency=False,
            drop_2023=False, ensemble_with=None, label="A"):
    feat_cols = NUMERIC + ["venue_enc"]
    if not drop_team_id:
        feat_cols = NUMERIC + [c+"_enc" for c in CATS]
    rows, correct = [], 0
    for _, r in test.iterrows():
        train = df[df["date"] < r["date"]]
        if drop_2023:
            train = train[train["season"].astype(str) != "2023"]
        weights = None
        if light_recency:
            weights = train["season"].astype(str).map(LIGHT_WEIGHTS).fillna(1.0).values
        p1 = fit_predict(train, r, feat_cols, weights)
        if ensemble_with is not None:
            p1 = (p1 + ensemble_with[r["match_id"]]) / 2
        pred = r["team1"] if p1 >= 0.5 else r["team2"]
        actual = matches.loc[matches["match_id"]==r["match_id"], "winner"].iloc[0]
        ok = pred == actual
        if ok: correct += 1
        rows.append({
            "match_id": r["match_id"], "date": str(r["date"].date()),
            "match": f"{r['team1'][:3].upper()} vs {r['team2'][:3].upper()}",
            "p1": round(p1, 2), "pred": pred, "actual": actual, "ok": ok,
        })
    print(f"\n=== {label}: {correct}/{len(test)} = {correct/len(test):.0%} ===")
    for x in rows:
        sym = "✓" if x["ok"] else "✗"
        print(f"  {x['date']}  {x['match']:<11}  P={x['p1']:.2f}  pred={x['pred'][:25]:<25}  actual={x['actual'][:25]:<25} {sym}")
    return correct, {x["match_id"]: x["p1"] for x in rows}


def main() -> None:
    df, matches = load()
    completed = df[df["season"].astype(str)=="2026"].sort_values("date")
    test = completed.tail(10)

    a, p_a = variant(df, matches, test, label="A · BASELINE")
    b, _   = variant(df, matches, test, light_recency=True, label="B · LIGHT RECENCY")
    c, p_c = variant(df, matches, test, drop_team_id=True, label="C · DROP TEAM IDENTITY")
    d, _   = variant(df, matches, test, drop_2023=True, label="D · DROP 2023 DATA")
    e, _   = variant(df, matches, test, drop_team_id=True, drop_2023=True, label="E · C + D")
    f, p_f = variant(df, matches, test, drop_team_id=True, ensemble_with=p_a, label="F · ENSEMBLE (A + C)")
    # G: average ensemble with A again -> weighted (0.75 A + 0.25 C)
    g, _   = variant(df, matches, test, drop_team_id=False, ensemble_with=p_f, label="G · ENSEMBLE + A (weighted)")

    print("\n" + "="*60)
    print("SUMMARY · last 10 completed IPL 2026 matches")
    print("="*60)
    print(f"A · Baseline                  {a}/10 = {a*10}%")
    print(f"B · Light recency             {b}/10 = {b*10}%")
    print(f"C · Drop team identity        {c}/10 = {c*10}%")
    print(f"D · Drop 2023 data            {d}/10 = {d*10}%")
    print(f"E · C + D combined            {e}/10 = {e*10}%")
    print(f"F · Ensemble (A + C)          {f}/10 = {f*10}%")
    print(f"G · Ensemble + A (0.75 A)     {g}/10 = {g*10}%")


if __name__ == "__main__":
    main()
