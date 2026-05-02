"""20 fresh variations exploring different factor combinations.

Each variation isolates a different angle:
  - Feature subset (form-only, pitch-only, toss-only, weather-only, etc.)
  - Model class (Stacking, Voting, MLP, SVM)
  - Training-data slice (2026-only, late-season-only)
  - Random-seed stability check
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

INIT = {"Mumbai Indians":"MI","Chennai Super Kings":"CSK","Royal Challengers Bengaluru":"RCB",
        "Kolkata Knight Riders":"KKR","Sunrisers Hyderabad":"SRH","Delhi Capitals":"DC",
        "Punjab Kings":"PBKS","Rajasthan Royals":"RR","Lucknow Super Giants":"LSG","Gujarat Titans":"GT"}
CATS = ["team1","team2","venue"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PROC/"features.csv", parse_dates=["date"]).dropna(subset=["target"])
    df = df.sort_values("date").reset_index(drop=True)
    venue = pd.read_csv(PROC/"venue_features.csv")
    ks = pd.read_csv(PROC/"ks_features.csv")
    ext = pd.read_csv(PROC/"external_features.csv")
    df = df.merge(venue, on="match_id", how="left")
    overlap = [c for c in ks.columns if c in df.columns and c != "match_id"]
    df = df.merge(ks.drop(columns=overlap), on="match_id", how="left")
    df = df.merge(ext, on="match_id", how="left")
    matches = pd.read_csv(PROC/"matches.csv", parse_dates=["date"])
    for c in CATS:
        le = LabelEncoder(); le.fit(df[c].astype(str))
        df[c+"_enc"] = le.transform(df[c].astype(str))
    return df, matches


def predict_one(model, train, row, cols, scale=False):
    X_train = train[cols].values
    sc = None
    if scale:
        sc = StandardScaler().fit(X_train); X_train = sc.transform(X_train)
    model.fit(X_train, train["target"])
    X = row[cols].values.reshape(1, -1)
    if scale: X = sc.transform(X)
    return float(model.predict_proba(X)[0, 1])


def run(df, matches, test, label, *, cols, model_factory, scale=False, train_filter=None):
    rows, correct = [], 0
    for _, r in test.iterrows():
        train = df[df["date"] < r["date"]].dropna(subset=cols)
        if train_filter is not None: train = train_filter(train, r)
        if len(train) < 20:  # safety
            train = df[df["date"] < r["date"]].dropna(subset=cols)
        m = model_factory()
        try:
            p1 = predict_one(m, train, r, cols, scale=scale)
        except Exception as e:
            p1 = 0.5
        actual = matches.loc[matches["match_id"]==r["match_id"], "winner"].iloc[0]
        pred = r["team1"] if p1 >= 0.5 else r["team2"]
        ok = pred == actual
        if ok: correct += 1
        rows.append({"p1": p1, "pred": pred, "actual": actual, "ok": ok})
    return label, correct, rows


def main() -> None:
    df, matches = load()
    test = df[df["season"].astype(str)=="2026"].sort_values("date").tail(10)

    # Feature buckets
    FORM    = ["t1_form","t2_form"]
    SCORING = ["t1_avg_scored","t2_avg_scored","t1_avg_conceded","t2_avg_conceded"]
    H2H     = ["h2h_t1_winrate","t1_venue_winrate","t2_venue_winrate"]
    TOSS    = ["t1_toss_winrate_10","t2_toss_winrate_10","toss_winner_is_t1","toss_decision_bat"]
    PITCH   = ["venue_avg_1st_inn","venue_avg_total","venue_runs_per_wkt",
               "venue_sixes_per_match","venue_bat_first_winrate","venue_pp_er","venue_death_er"]
    PHASE   = ["t1_pp_rr_for","t2_pp_rr_for","t1_md_rr_for","t2_md_rr_for",
               "t1_de_rr_for","t2_de_rr_for","t1_de_rr_against","t2_de_rr_against"]
    WEATHER = ["temp_c","humidity","wind_kmh","precip_mm","dew_score","is_summer"]
    GROUND  = ["alt_m","boundary_m","pitch_type"]
    CONTEXT = ["t1_days_since_last","t2_days_since_last","t1_is_home_venue","t2_is_home_venue",
               "t1_win_streak","t2_win_streak","t1_avg_margin","t2_avg_margin"]
    IDS     = ["team1_enc","team2_enc","venue_enc"]

    def gb(d=3, n=400, lr=0.05): return lambda: GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42)
    def gb_seed(seed): return lambda: GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=seed)
    def rf(d=8): return lambda: RandomForestClassifier(n_estimators=400, max_depth=d, random_state=42)
    def lr_(C=0.5): return lambda: LogisticRegression(max_iter=2000, C=C)
    def mlp(): return lambda: MLPClassifier(hidden_layer_sizes=(32,16), max_iter=600, random_state=42)
    def svm(): return lambda: SVC(probability=True, kernel="rbf", C=1.0, random_state=42)

    def stacker():
        return lambda: StackingClassifier(
            estimators=[("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)),
                        ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))],
            final_estimator=LogisticRegression(max_iter=1000), cv=3,
        )

    def voter():
        return lambda: VotingClassifier(
            estimators=[("gb", GradientBoostingClassifier(n_estimators=400, max_depth=3, random_state=42)),
                        ("rf", RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42)),
                        ("lr", LogisticRegression(max_iter=2000))],
            voting="soft",
        )

    season_2026_only = lambda train, _r: train[train["season"].astype(str) == "2026"]
    late_season_only = lambda train, _r: train[train["t1_season_match_no"] >= 3] if "t1_season_match_no" in train.columns else train

    variations = [
        # Feature subsets
        ("V01 · FORM only (2 feats, GB)",                  dict(cols=FORM + IDS,                model_factory=gb())),
        ("V02 · SCORING only (4 feats, GB)",               dict(cols=SCORING + IDS,             model_factory=gb())),
        ("V03 · TOSS only (4 feats, GB)",                  dict(cols=TOSS + IDS,                model_factory=gb())),
        ("V04 · H2H + venue only (3 feats, GB)",           dict(cols=H2H + IDS,                 model_factory=gb())),
        ("V05 · PITCH only (7 feats, GB)",                 dict(cols=PITCH + IDS,               model_factory=gb())),
        ("V06 · PHASE-rates only (8 feats, GB)",           dict(cols=PHASE + IDS,               model_factory=gb())),
        ("V07 · WEATHER only (6 feats, GB)",               dict(cols=WEATHER + IDS,             model_factory=gb())),
        ("V08 · GROUND attrs only (3 feats, GB)",          dict(cols=GROUND + IDS,              model_factory=gb())),
        ("V09 · CONTEXT only (8 feats, GB)",               dict(cols=CONTEXT + IDS,             model_factory=gb())),
        ("V10 · ALL minus team_enc (no brand)",            dict(cols=FORM+SCORING+H2H+TOSS+PITCH+PHASE+WEATHER+GROUND+CONTEXT+["venue_enc"], model_factory=gb())),
        # Model class variants on the FORM+SCORING+H2H+TOSS bundle
        ("V11 · MLP neural net (full)",                    dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=mlp(),    scale=True)),
        ("V12 · SVM RBF (full)",                           dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=svm(),    scale=True)),
        ("V13 · Stacking (GB+RF→LR)",                      dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=stacker())),
        ("V14 · Voting (GB+RF+LR soft)",                   dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=voter(),  scale=True)),
        # Seed-stability check
        ("V15 · GB seed=42 (default)",                     dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=gb_seed(42))),
        ("V16 · GB seed=7",                                dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=gb_seed(7))),
        ("V17 · GB seed=99",                               dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=gb_seed(99))),
        # Training-data scope variants
        ("V18 · Train 2026-only",                          dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=gb(), train_filter=season_2026_only)),
        ("V19 · Drop early-season (>=3 played)",           dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=gb(), train_filter=late_season_only)),
        ("V20 · No identities (pure stats)",               dict(cols=FORM+SCORING+H2H+TOSS,     model_factory=gb())),
    ]

    print(f"\nRunning {len(variations)} variations on last 10 IPL 2026 matches...\n")
    results = []
    for label, kwargs in variations:
        results.append(run(df, matches, test, label, **kwargs))

    sorted_r = sorted(results, key=lambda x: -x[1])
    print("=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    for label, c, _ in sorted_r:
        bar = "█" * c + "░" * (10 - c)
        print(f"{label:<48} {bar}  {c}/10 = {c*10}%")

    # Match #8 spotlight (CSK v GT — the unanimous miss)
    print(f"\nWhich variants got CSK v GT right (match index 7)?")
    for label, _, rows in results:
        # find CSK v GT (april 26)
        for i, ((_, r), row) in enumerate(zip(test.iterrows(), rows)):
            if r["team1"]=="Chennai Super Kings" and r["team2"]=="Gujarat Titans":
                if row["ok"]:
                    print(f"  ✓ {label}  (P={row['p1']:.2f})")


if __name__ == "__main__":
    main()
