"""Full feature-set evaluation across IPL 2024, 2025, and last 10 of 2026.

Combines:
  - Original 13 numeric features (form, scoring, h2h, toss)
  - 7 venue/pitch features
  - 41 kitchen-sink features (phase splits, streaks, margins, rest, home, etc.)
  - Categorical encodings (team1, team2, venue)

Reports A · Baseline-now and G · Ensemble (75/25) accuracy.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ORIG_NUMERIC = ["t1_form","t2_form","t1_avg_scored","t2_avg_scored",
                "t1_avg_conceded","t2_avg_conceded","t1_venue_winrate","t2_venue_winrate",
                "h2h_t1_winrate","t1_toss_winrate_10","t2_toss_winrate_10",
                "toss_winner_is_t1","toss_decision_bat"]
PITCH = ["venue_avg_1st_inn","venue_avg_total","venue_runs_per_wkt",
         "venue_sixes_per_match","venue_bat_first_winrate",
         "venue_pp_er","venue_death_er"]
CATS = ["team1","team2","venue"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    venue = pd.read_csv(PROC / "venue_features.csv")
    ks = pd.read_csv(PROC / "ks_features.csv")
    df = df.merge(venue, on="match_id", how="left")
    overlap = [c for c in ks.columns if c in df.columns and c != "match_id"]
    ks = ks.drop(columns=overlap)
    df = df.merge(ks, on="match_id", how="left")
    ext = pd.read_csv(PROC / "external_features.csv")
    df = df.merge(ext, on="match_id", how="left")
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    for c in CATS:
        le = LabelEncoder(); le.fit(df[c].astype(str))
        df[c+"_enc"] = le.transform(df[c].astype(str))
    ks_cols = [c for c in ks.columns if c != "match_id"]  # already deduped above
    return df, matches, ks_cols


def evaluate(df, matches, test, *, full_cols, no_brand_cols, label):
    a_correct = g_correct = 0
    for _, r in test.iterrows():
        train = df[df["date"] < r["date"]]
        m_full = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
        m_full.fit(train[full_cols], train["target"])
        m_nb = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
        m_nb.fit(train[no_brand_cols], train["target"])
        X_full = pd.DataFrame([r[full_cols].values], columns=full_cols)
        X_nb = pd.DataFrame([r[no_brand_cols].values], columns=no_brand_cols)
        p_a = float(m_full.predict_proba(X_full)[0,1])
        p_c = float(m_nb.predict_proba(X_nb)[0,1])
        p_g = 0.75 * p_a + 0.25 * p_c
        actual = matches.loc[matches["match_id"] == r["match_id"], "winner"].iloc[0]
        if (r["team1"] if p_a >= 0.5 else r["team2"]) == actual: a_correct += 1
        if (r["team1"] if p_g >= 0.5 else r["team2"]) == actual: g_correct += 1
    n = len(test)
    print(f"{label:<32}  A {a_correct:>3}/{n} = {a_correct/n:.0%}   G {g_correct:>3}/{n} = {g_correct/n:.0%}")
    return a_correct, g_correct, n


def main() -> None:
    df, matches, ks_cols = load()
    EXT = ["alt_m","boundary_m","pitch_type","temp_c","humidity","wind_kmh",
           "precip_mm","dew_score","is_summer"]
    full_cols = ORIG_NUMERIC + PITCH + ks_cols + EXT + [c+"_enc" for c in CATS]
    no_brand_cols = ORIG_NUMERIC + PITCH + ks_cols + EXT + ["venue_enc"]
    print(f"Total numeric features: {len(full_cols)}\n")

    seasons = {
        "IPL 2024 (full)":  df[df["season"].astype(str)=="2024"],
        "IPL 2025 (full)":  df[df["season"].astype(str)=="2025"],
        "IPL 2026 (full)":  df[df["season"].astype(str)=="2026"],
        "IPL 2026 last 10": df[df["season"].astype(str)=="2026"].tail(10),
    }
    print(f"{'window':<32}  {'baseline A':>14}    {'ensemble G':>14}")
    print("-"*70)
    for label, test in seasons.items():
        evaluate(df, matches, test, full_cols=full_cols, no_brand_cols=no_brand_cols, label=label)


if __name__ == "__main__":
    main()
