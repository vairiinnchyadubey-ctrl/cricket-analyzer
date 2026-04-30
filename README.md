# IPL Match Winner Predictor

ML model that predicts IPL match winners using ball-by-ball data from the last 3 seasons (2023-2025) plus the live 2026 season.

## Pipeline

```
Cricsheet zip  →  src/scrape.py    →  data/raw/*.json
                  src/parse.py     →  data/processed/{matches,deliveries}.csv
                  src/features.py  →  data/processed/features.csv
                  src/train.py     →  models/winner_model.joblib
                  src/predict.py   →  predictions for upcoming matches
```

## Setup

```bash
pip3 install -r requirements.txt
```

## Run end-to-end

```bash
python3 -m src.scrape         # download IPL JSONs (filters to 2023-2026)
python3 -m src.parse          # flatten to CSVs
python3 -m src.features       # build leak-free per-match features
python3 -m src.train          # train + evaluate (train: 2023-24, test: 2025+)
```

## Predict an upcoming match

```bash
python3 -m src.predict \
  --team1 "Mumbai Indians" \
  --team2 "Chennai Super Kings" \
  --venue "Wankhede Stadium, Mumbai" \
  --toss-winner "Mumbai Indians" \
  --toss-decision bat
```

## Features used

- Team rolling form (last 5 matches)
- Avg runs scored / conceded
- Venue-specific win rate
- Head-to-head win rate
- Toss winner & decision
- Encoded team / venue identity

## Evaluation

Train: 2023 + 2024 seasons. Test: 2025 (and 2026 as it plays out).
Metrics reported: accuracy, log-loss, Brier score.

To track live 2026 accuracy, re-run `scrape → parse → features` weekly and log
predictions vs. actuals.
