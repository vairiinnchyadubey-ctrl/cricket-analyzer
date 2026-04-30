"""Walk-forward A vs G comparison on the FIRST 20 matches of IPL 2025.

For each test match, models train on every match strictly before it
(IPL 2023 + 2024 + earlier 2025 matches).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

NUMERIC = ["t1_form","t2_form","t1_avg_scored","t2_avg_scored",
           "t1_avg_conceded","t2_avg_conceded","t1_venue_winrate","t2_venue_winrate",
           "h2h_t1_winrate","t1_toss_winrate_10","t2_toss_winrate_10",
           "toss_winner_is_t1","toss_decision_bat"]
CATS = ["team1","team2","venue"]
SKIP = 0
N_TEST = 999  # every match


def main() -> None:
    df = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    venue_feats = pd.read_csv(PROC / "venue_features.csv")
    df = df.merge(venue_feats, on="match_id", how="left")
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    for c in CATS:
        le = LabelEncoder(); le.fit(df[c].astype(str))
        df[c+"_enc"] = le.transform(df[c].astype(str))

    PITCH = ["venue_avg_1st_inn","venue_avg_total","venue_runs_per_wkt",
             "venue_sixes_per_match","venue_bat_first_winrate",
             "venue_pp_er","venue_death_er"]
    full_cols = NUMERIC + PITCH + [c+"_enc" for c in CATS]
    no_brand_cols = NUMERIC + PITCH + ["venue_enc"]

    season_2025 = df[df["season"].astype(str) == "2025"].sort_values("date").reset_index(drop=True)
    test = season_2025.iloc[SKIP:SKIP + N_TEST]
    print(f"IPL 2025 matches {SKIP+1}–{SKIP+N_TEST} "
          f"({test['date'].min().date()} → {test['date'].max().date()})\n")

    rows_a, rows_g = [], []
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
        pa = r["team1"] if p_a >= 0.5 else r["team2"]
        pg = r["team1"] if p_g >= 0.5 else r["team2"]
        ok_a = pa == actual; ok_g = pg == actual
        if ok_a: a_correct += 1
        if ok_g: g_correct += 1

        match_lbl = f"{r['team1'][:3].upper()} vs {r['team2'][:3].upper()}"
        rows_a.append((r["date"].date(), match_lbl, p_a, pa, actual, ok_a))
        rows_g.append((r["date"].date(), match_lbl, p_g, pg, actual, ok_g))

    print(f"{'#':<3}{'date':<12}{'match':<14}{'A·P':>5}  {'A pick':<25}{'G·P':>5}  {'G pick':<25}{'actual':<25}{'A':<3}{'G':<3}")
    for i, ((d, lbl, pa, pa_pick, actual, oa), (_, _, pg, pg_pick, _, og)) in enumerate(zip(rows_a, rows_g), 1):
        sa = "✓" if oa else "✗"; sg = "✓" if og else "✗"
        print(f"{i:<3}{str(d):<12}{lbl:<14}{pa:>5.2f}  {pa_pick[:25]:<25}{pg:>5.2f}  {pg_pick[:25]:<25}{actual[:25]:<25}{sa:<3}{sg:<3}")

    print()
    print("="*60)
    n = len(test)
    print(f"A · Baseline                  {a_correct}/{n} = {a_correct/n:.0%}")
    print(f"G · Ensemble (75% A + 25% C)  {g_correct}/{n} = {g_correct/n:.0%}")
    diff = g_correct - a_correct
    print(f"\nG vs A: {'+' if diff > 0 else ''}{diff} matches "
          f"({'G better' if diff > 0 else 'A better' if diff < 0 else 'tied'})")


if __name__ == "__main__":
    main()
