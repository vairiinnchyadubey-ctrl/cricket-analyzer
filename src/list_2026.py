"""List every 2026 match: predicted vs actual."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ORIG = ["t1_form","t2_form","t1_avg_scored","t2_avg_scored",
        "t1_avg_conceded","t2_avg_conceded","t1_venue_winrate","t2_venue_winrate",
        "h2h_t1_winrate","t1_toss_winrate_10","t2_toss_winrate_10",
        "toss_winner_is_t1","toss_decision_bat"]
PITCH = ["venue_avg_1st_inn","venue_avg_total","venue_runs_per_wkt",
         "venue_sixes_per_match","venue_bat_first_winrate","venue_pp_er","venue_death_er"]
EXT   = ["alt_m","boundary_m","pitch_type","temp_c","humidity","wind_kmh",
         "precip_mm","dew_score","is_summer"]
CATS = ["team1","team2","venue"]
INIT = {"Mumbai Indians":"MI","Chennai Super Kings":"CSK","Royal Challengers Bengaluru":"RCB",
        "Kolkata Knight Riders":"KKR","Sunrisers Hyderabad":"SRH","Delhi Capitals":"DC",
        "Punjab Kings":"PBKS","Rajasthan Royals":"RR","Lucknow Super Giants":"LSG","Gujarat Titans":"GT"}


def main() -> None:
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
    full_cols = ORIG + PITCH + ks_cols + EXT + [c+"_enc" for c in CATS]

    test = df[df["season"].astype(str)=="2026"].sort_values("date")
    correct = 0
    print(f"{'#':<4}{'date':<12}{'match':<14}{'P':>5}  {'predicted':<6}  {'actual':<6}  result")
    print("-"*60)
    for i, (_, r) in enumerate(test.iterrows(), 1):
        train = df[df["date"] < r["date"]]
        m = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)
        m.fit(train[full_cols], train["target"])
        X = pd.DataFrame([r[full_cols].values], columns=full_cols)
        p1 = float(m.predict_proba(X)[0,1])
        pred = r["team1"] if p1 >= 0.5 else r["team2"]
        actual = matches.loc[matches["match_id"]==r["match_id"],"winner"].iloc[0]
        ok = pred == actual
        if ok: correct += 1
        sym = "✓" if ok else "✗"
        match_str = f"{INIT.get(r['team1'],r['team1'][:3]):<4} v {INIT.get(r['team2'],r['team2'][:3]):<4}"
        print(f"{i:<4}{str(r['date'].date()):<12}{match_str:<14}{p1:>5.2f}  {INIT.get(pred,pred[:6]):<6}  {INIT.get(actual,actual[:6]):<6}  {sym}")

    print("-"*60)
    print(f"\nFinal: {correct}/{len(test)} = {correct/len(test):.1%}")


if __name__ == "__main__":
    main()
