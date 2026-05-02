"""Replicate the Claws333/ipl-2025-predictor method on our REAL IPL data.

Their model: RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced')
Their features: all diffs (team1 stat - team2 stat)

Two evaluations:
  1. THEIR method: train on full dataset, evaluate in-sample (their reported metric).
     Demonstrates how 85%-style numbers come from leakage.
  2. OUR method: walk-forward (train on past matches only, predict future). Honest test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

INIT = {"Mumbai Indians":"MI","Chennai Super Kings":"CSK","Royal Challengers Bengaluru":"RCB",
        "Kolkata Knight Riders":"KKR","Sunrisers Hyderabad":"SRH","Delhi Capitals":"DC",
        "Punjab Kings":"PBKS","Rajasthan Royals":"RR","Lucknow Super Giants":"LSG","Gujarat Titans":"GT"}


def main() -> None:
    df = pd.read_csv(PROC/"features.csv", parse_dates=["date"]).dropna(subset=["target"])
    df = df.sort_values("date").reset_index(drop=True)
    venue = pd.read_csv(PROC/"venue_features.csv")
    ks = pd.read_csv(PROC/"ks_features.csv")
    df = df.merge(venue, on="match_id", how="left")
    overlap = [c for c in ks.columns if c in df.columns and c != "match_id"]
    df = df.merge(ks.drop(columns=overlap), on="match_id", how="left")
    matches = pd.read_csv(PROC/"matches.csv", parse_dates=["date"])

    # Build "diff" features (their style)
    df["form_diff"]      = df["t1_form"] - df["t2_form"]
    df["scored_diff"]    = df["t1_avg_scored"] - df["t2_avg_scored"]
    df["conceded_diff"]  = df["t2_avg_conceded"] - df["t1_avg_conceded"]   # lower-is-better -> flip
    df["venue_diff"]     = df["t1_venue_winrate"] - df["t2_venue_winrate"]
    df["toss_winrate_diff"] = df["t1_toss_winrate_10"] - df["t2_toss_winrate_10"]
    df["pp_for_diff"]    = df["t1_pp_rr_for"] - df["t2_pp_rr_for"]
    df["pp_against_diff"]= df["t2_pp_rr_against"] - df["t1_pp_rr_against"]
    df["md_for_diff"]    = df["t1_md_rr_for"] - df["t2_md_rr_for"]
    df["death_for_diff"] = df["t1_de_rr_for"] - df["t2_de_rr_for"]
    df["death_econ_diff"]= df["t2_de_rr_against"] - df["t1_de_rr_against"]
    df["wkts_taken_diff"]= df["t1_wickets_taken_5"] - df["t2_wickets_taken_5"]
    df["wkts_lost_diff"] = df["t2_wickets_lost_5"] - df["t1_wickets_lost_5"]
    df["streak_diff"]    = df["t1_win_streak"] - df["t2_win_streak"]
    df["margin_diff"]    = df["t1_avg_margin"] - df["t2_avg_margin"]
    df["toss_bonus"]     = df["toss_winner_is_t1"] * (1.0 - df["venue_bat_first_winrate"])  # toss × chase bias
    df["chase_bias"]     = 1.0 - df["venue_bat_first_winrate"]
    df["h2h_diff"]       = df["h2h_t1_winrate"]

    feat_cols = [
        "form_diff","scored_diff","conceded_diff","venue_diff","toss_winrate_diff",
        "pp_for_diff","pp_against_diff","md_for_diff","death_for_diff","death_econ_diff",
        "wkts_taken_diff","wkts_lost_diff","streak_diff","margin_diff",
        "toss_bonus","chase_bias","h2h_diff",
    ]

    print(f"Replicating Claws333 method · {len(feat_cols)} diff features · RF(n=500, depth=15)\n")

    # =========================================================================
    # 1. Their method: in-sample (leakage demo)
    # =========================================================================
    print("=" * 60)
    print("1 · THEIR METHOD: train + test on full dataset (their reported style)")
    print("=" * 60)
    full_df = df.dropna(subset=feat_cols)
    rf = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced")
    rf.fit(full_df[feat_cols], full_df["target"])
    pred = rf.predict(full_df[feat_cols])
    in_sample_acc = (pred == full_df["target"]).mean()
    print(f"In-sample accuracy on {len(full_df)} matches: {in_sample_acc:.1%}")
    print("⚠️  This is LEAKAGE — model was graded on the same data it learned from.")

    # =========================================================================
    # 2. Walk-forward (our honest method)
    # =========================================================================
    print("\n" + "=" * 60)
    print("2 · WALK-FORWARD: train on past matches only, predict last 10")
    print("=" * 60)

    test = full_df[full_df["season"].astype(str)=="2026"].sort_values("date").tail(10)
    correct = 0
    rows = []
    for _, r in test.iterrows():
        train = full_df[full_df["date"] < r["date"]]
        rf2 = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42, class_weight="balanced")
        rf2.fit(train[feat_cols], train["target"])
        X = pd.DataFrame([r[feat_cols].values], columns=feat_cols)
        p1 = float(rf2.predict_proba(X)[0,1])
        actual = matches.loc[matches["match_id"]==r["match_id"],"winner"].iloc[0]
        pred_team = r["team1"] if p1 >= 0.5 else r["team2"]
        ok = pred_team == actual
        if ok: correct += 1
        rows.append((r["date"].date(), r["team1"], r["team2"], p1, pred_team, actual, ok))

    print(f"\n{'#':<3}{'date':<12}{'match':<12}{'P(t1)':>6}  {'pred':<6}  {'actual':<6} res")
    for i, (d, t1, t2, p, pred, act, ok) in enumerate(rows, 1):
        sym = "✓" if ok else "✗"
        m = f"{INIT.get(t1,t1[:3])}v{INIT.get(t2,t2[:3])}"
        print(f"{i:<3}{str(d):<12}{m:<12}{p:>6.2f}  {INIT.get(pred,pred[:6]):<6}  {INIT.get(act,act[:6]):<6} {sym}")

    print(f"\n→ Walk-forward accuracy: {correct}/10 = {correct/10:.0%}")
    print(f"\nGap between leakage-style ({in_sample_acc:.0%}) and honest ({correct/10:.0%}) "
          f"= {(in_sample_acc - correct/10)*100:.0f} percentage points.")
    print("That gap IS the leakage.")


if __name__ == "__main__":
    main()
