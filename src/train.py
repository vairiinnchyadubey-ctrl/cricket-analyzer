"""Train winner-prediction model with time-based split.

Train on 2023-2024, test on 2025 (and 2026 if present).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder

HAVE_XGB = False
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    pass
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

NUMERIC_FEATURES = [
    "t1_form", "t2_form",
    "t1_avg_scored", "t2_avg_scored",
    "t1_avg_conceded", "t2_avg_conceded",
    "t1_venue_winrate", "t2_venue_winrate",
    "h2h_t1_winrate",
    "toss_winner_is_t1", "toss_decision_bat",
]
CAT_FEATURES = ["team1", "team2", "venue"]


def main() -> None:
    df = pd.read_csv(PROC_DIR / "features.csv", parse_dates=["date"])
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    encoders = {}
    for c in CAT_FEATURES:
        le = LabelEncoder()
        df[c + "_enc"] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    feature_cols = NUMERIC_FEATURES + [c + "_enc" for c in CAT_FEATURES]

    train_mask = df["season"].astype(str).isin(["2023", "2024"])
    test_mask = df["season"].astype(str).isin(["2025", "2026"])
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, "target"]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, "target"]

    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit("Empty split — re-check seasons in features.csv")

    if HAVE_XGB:
        model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            eval_metric="logloss",
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
        )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n=== Test metrics (2025+) ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"LogLoss  : {log_loss(y_test, proba):.3f}")
    print(f"Brier    : {brier_score_loss(y_test, proba):.3f}")
    print(classification_report(y_test, pred, digits=3))

    # Feature importance
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nTop features:")
        print(imp.head(10).to_string())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "encoders": encoders, "feature_cols": feature_cols},
                MODEL_DIR / "winner_model.joblib")
    print(f"\nSaved model -> {MODEL_DIR / 'winner_model.joblib'}")


if __name__ == "__main__":
    main()
