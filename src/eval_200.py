"""Run top variants from eval_factors.py on the LAST 200 matches (walk-forward)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
CATS = ["team1","team2","venue"]
N_TEST = 200


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PROC/"features.csv", parse_dates=["date"]).dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
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


def run(df, matches, test, label, *, cols, model_factory, scale=False):
    correct = 0
    for _, r in tqdm(test.iterrows(), total=len(test), desc=label[:30], leave=False):
        train = df[df["date"] < r["date"]].dropna(subset=cols)
        if len(train) < 30:
            continue
        m = model_factory()
        try:
            p1 = predict_one(m, train, r, cols, scale=scale)
        except Exception:
            p1 = 0.5
        actual = matches.loc[matches["match_id"]==r["match_id"], "winner"].iloc[0]
        pred = r["team1"] if p1 >= 0.5 else r["team2"]
        if pred == actual: correct += 1
    return label, correct, len(test)


def main() -> None:
    df, matches = load()
    test = df.tail(N_TEST)
    print(f"Test window: last {N_TEST} matches ({test['date'].min().date()} → {test['date'].max().date()})\n")

    FORM    = ["t1_form","t2_form"]
    SCORING = ["t1_avg_scored","t2_avg_scored","t1_avg_conceded","t2_avg_conceded"]
    H2H     = ["h2h_t1_winrate","t1_venue_winrate","t2_venue_winrate"]
    TOSS    = ["t1_toss_winrate_10","t2_toss_winrate_10","toss_winner_is_t1","toss_decision_bat"]
    IDS     = ["team1_enc","team2_enc","venue_enc"]
    PHASE   = ["t1_pp_rr_for","t2_pp_rr_for","t1_de_rr_for","t2_de_rr_for","t1_de_rr_against","t2_de_rr_against"]
    PITCH   = ["venue_avg_1st_inn","venue_runs_per_wkt","venue_bat_first_winrate"]

    def gb(seed=42): return lambda: GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=seed)
    def mlp(): return lambda: MLPClassifier(hidden_layer_sizes=(32,16), max_iter=600, random_state=42)
    def stacker(): return lambda: StackingClassifier(
        estimators=[("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))],
        final_estimator=LogisticRegression(max_iter=1000), cv=3,
    )
    def voter(): return lambda: VotingClassifier(
        estimators=[("gb", GradientBoostingClassifier(n_estimators=400, max_depth=3, random_state=42)),
                    ("rf", RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42)),
                    ("lr", LogisticRegression(max_iter=2000))],
        voting="soft",
    )

    variations = [
        ("V01 · GB-16 baseline (form+scoring+h2h+toss+IDs)",  dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=gb())),
        ("V02 · GB-7 SCORING-only",                            dict(cols=SCORING+IDS,                model_factory=gb())),
        ("V03 · GB-22 + PHASE + PITCH",                        dict(cols=FORM+SCORING+H2H+TOSS+PHASE+PITCH+IDS, model_factory=gb())),
        ("V04 · GB seed=7  (stability check)",                 dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=gb(7))),
        ("V05 · GB seed=99 (stability check)",                 dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=gb(99))),
        ("V06 · MLP neural net",                               dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=mlp(),    scale=True)),
        ("V07 · Stacking (GB+RF→LR)",                          dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=stacker())),
        ("V08 · Voting (GB+RF+LR soft)",                       dict(cols=FORM+SCORING+H2H+TOSS+IDS, model_factory=voter(),  scale=True)),
    ]

    print(f"Running {len(variations)} variants on {N_TEST} matches...\n")
    results = []
    for label, kwargs in variations:
        results.append(run(df, matches, test, label, **kwargs))

    sorted_r = sorted(results, key=lambda x: -x[1])
    print("\n" + "=" * 60)
    print(f"LEADERBOARD · last {N_TEST} matches walk-forward")
    print("=" * 60)
    for label, c, n in sorted_r:
        bar = "█" * int(c / n * 20) + "░" * (20 - int(c / n * 20))
        print(f"{label:<46} {bar}  {c}/{n} = {c/n:.1%}")


if __name__ == "__main__":
    main()
