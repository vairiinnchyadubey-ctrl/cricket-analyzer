"""Run 20 model variations on the LAST 10 IPL 2026 matches and rank them."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ORIG = ["t1_form","t2_form","t1_avg_scored","t2_avg_scored",
        "t1_avg_conceded","t2_avg_conceded","t1_venue_winrate","t2_venue_winrate",
        "h2h_t1_winrate","t1_toss_winrate_10","t2_toss_winrate_10",
        "toss_winner_is_t1","toss_decision_bat"]
PITCH = ["venue_avg_1st_inn","venue_avg_total","venue_runs_per_wkt",
         "venue_sixes_per_match","venue_bat_first_winrate","venue_pp_er","venue_death_er"]
EXT = ["alt_m","boundary_m","pitch_type","temp_c","humidity","wind_kmh",
       "precip_mm","dew_score","is_summer"]
CATS = ["team1","team2","venue"]
INIT = {"Mumbai Indians":"MI","Chennai Super Kings":"CSK","Royal Challengers Bengaluru":"RCB",
        "Kolkata Knight Riders":"KKR","Sunrisers Hyderabad":"SRH","Delhi Capitals":"DC",
        "Punjab Kings":"PBKS","Rajasthan Royals":"RR","Lucknow Super Giants":"LSG","Gujarat Titans":"GT"}
RECENCY = {"2023":0.7, "2024":1.0, "2025":1.5, "2026":2.0}


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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
    ks_cols = [c for c in ks.columns if c != "match_id" and c not in overlap]
    return df, matches, ks_cols


def predict_one(model, train, row, cols, weights=None, scale=False):
    X_train = train[cols].values
    if scale:
        sc = StandardScaler().fit(X_train); X_train = sc.transform(X_train)
    try:
        model.fit(X_train, train["target"], sample_weight=weights)
    except TypeError:
        model.fit(X_train, train["target"])
    X = row[cols].values.reshape(1, -1)
    if scale: X = sc.transform(X)
    return float(model.predict_proba(X)[0,1])


def run_variation(df, matches, test, label, *, cols, drop_2023=False,
                  recency=False, model_factory, scale=False):
    rows, correct = [], 0
    for _, r in test.iterrows():
        train = df[df["date"] < r["date"]]
        if drop_2023:
            train = train[train["season"].astype(str) != "2023"]
        weights = None
        if recency:
            weights = train["season"].astype(str).map(RECENCY).fillna(1.0).values
        m = model_factory()
        p1 = predict_one(m, train, r, cols, weights=weights, scale=scale)
        actual = matches.loc[matches["match_id"]==r["match_id"],"winner"].iloc[0]
        pred = r["team1"] if p1 >= 0.5 else r["team2"]
        ok = pred == actual
        if ok: correct += 1
        rows.append({"match_id": r["match_id"], "p1": p1, "pred": pred, "actual": actual, "ok": ok})
    return label, correct, rows


def main() -> None:
    df, matches, ks_cols = load()
    test = df[df["season"].astype(str)=="2026"].sort_values("date").tail(10)

    cols_orig = ORIG + ["team1_enc","team2_enc","venue_enc"]
    cols_pitch = ORIG + PITCH + ["team1_enc","team2_enc","venue_enc"]
    cols_ks = ORIG + PITCH + ks_cols + ["team1_enc","team2_enc","venue_enc"]
    cols_full = ORIG + PITCH + ks_cols + EXT + ["team1_enc","team2_enc","venue_enc"]
    cols_no_brand = ORIG + PITCH + ks_cols + EXT + ["venue_enc"]

    def gb(n=400, d=3, lr=0.05): return lambda: GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr, random_state=42)
    def rf(n=400, d=8):           return lambda: RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
    def et(n=400, d=8):           return lambda: ExtraTreesClassifier(n_estimators=n, max_depth=d, random_state=42)
    def lr(C=0.5):                return lambda: LogisticRegression(max_iter=2000, C=C)
    def knn(k=5):                 return lambda: KNeighborsClassifier(n_neighbors=k)
    def hgb(d=None, lr=0.05):     return lambda: HistGradientBoostingClassifier(max_depth=d, learning_rate=lr, random_state=42)
    def nb():                     return lambda: GaussianNB()
    def gb_calibrated():          return lambda: CalibratedClassifierCV(GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42), cv=3, method="isotonic")

    variations = [
        ("V01 · GB-13 (baseline)",            dict(cols=cols_orig,     model_factory=gb())),
        ("V02 · GB-71 (full)",                dict(cols=cols_full,     model_factory=gb())),
        ("V03 · GB-13 lr=0.10",               dict(cols=cols_orig,     model_factory=gb(lr=0.10))),
        ("V04 · GB-13 lr=0.02 n=800",         dict(cols=cols_orig,     model_factory=gb(n=800, lr=0.02))),
        ("V05 · GB-13 depth=2",               dict(cols=cols_orig,     model_factory=gb(d=2))),
        ("V06 · GB-13 depth=5",               dict(cols=cols_orig,     model_factory=gb(d=5))),
        ("V07 · GB-13 n=200",                 dict(cols=cols_orig,     model_factory=gb(n=200))),
        ("V08 · GB-13 n=1000",                dict(cols=cols_orig,     model_factory=gb(n=1000))),
        ("V09 · GB-13 calibrated",            dict(cols=cols_orig,     model_factory=gb_calibrated())),
        ("V10 · HistGB-13",                   dict(cols=cols_orig,     model_factory=hgb())),
        ("V11 · GB-71 no-team-id",            dict(cols=cols_no_brand, model_factory=gb())),
        ("V12 · GB-71 recency-weighted",      dict(cols=cols_full,     model_factory=gb(), recency=True)),
        ("V13 · GB-71 drop-2023",             dict(cols=cols_full,     model_factory=gb(), drop_2023=True)),
        ("V14 · RF-400 d=8 (full)",           dict(cols=cols_full,     model_factory=rf())),
        ("V15 · RF-1000 d=12 (13)",           dict(cols=cols_orig,     model_factory=rf(n=1000, d=12))),
        ("V16 · ExtraTrees-400 (13)",         dict(cols=cols_orig,     model_factory=et())),
        ("V17 · LogReg C=0.5 (13)",           dict(cols=cols_orig,     model_factory=lr(0.5),  scale=True)),
        ("V18 · LogReg C=2 (full)",           dict(cols=cols_full,     model_factory=lr(2.0),  scale=True)),
        ("V19 · KNN k=15 (13)",               dict(cols=cols_orig,     model_factory=knn(15),  scale=True)),
        ("V20 · GaussianNB (13)",             dict(cols=cols_orig,     model_factory=nb(),     scale=True)),
    ]

    print(f"\nRunning {len(variations)} variations on last 10 IPL 2026 matches...\n")
    results = []
    for label, kwargs in variations:
        results.append(run_variation(df, matches, test, label, **kwargs))

    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: -x[1])

    print("="*60)
    print("LEADERBOARD · last 10 IPL 2026 matches")
    print("="*60)
    for label, c, _ in results_sorted:
        bar = "█" * c + "░" * (10 - c)
        print(f"{label:<40}  {bar}  {c}/10 = {c*10}%")

    # Per-match table for top 6
    top6 = results_sorted[:6]
    print(f"\nTop 6 side-by-side per match:\n")
    print(f"{'#':<3}{'date':<11}{'match':<10}", end="")
    for label, _, _ in top6:
        print(f"{label.split(' · ')[0]:<6} ", end="")
    print(f" {'actual':<6}")
    print("-" * (3 + 11 + 10 + 7*len(top6) + 8))
    for i, (_, r) in enumerate(test.iterrows(), 1):
        actual = matches.loc[matches["match_id"]==r["match_id"],"winner"].iloc[0]
        m_lbl = f"{INIT.get(r['team1'],r['team1'][:3])}v{INIT.get(r['team2'],r['team2'][:3])}"
        print(f"{i:<3}{str(r['date'].date()):<11}{m_lbl:<10}", end="")
        for label, _, rows in top6:
            row = rows[i-1]
            sym = "✓" if row["ok"] else "✗"
            print(f"{sym:<6} ", end="")
        print(f" {INIT.get(actual, actual[:6]):<6}")

    # Mega ensemble of top 5
    top5 = results_sorted[:5]
    mega_correct = 0
    for i in range(len(test)):
        p_avg = np.mean([rows[i]["p1"] for _, _, rows in top5])
        actual = top5[0][2][i]["actual"]
        team1 = matches.loc[matches["match_id"]==top5[0][2][i]["match_id"], "team1"].iloc[0]
        team2 = matches.loc[matches["match_id"]==top5[0][2][i]["match_id"], "team2"].iloc[0]
        pred = team1 if p_avg >= 0.5 else team2
        if pred == actual: mega_correct += 1
    print(f"\nV21 · ENSEMBLE of top 5 (averaged probabilities): {mega_correct}/10 = {mega_correct*10}%")


if __name__ == "__main__":
    main()
