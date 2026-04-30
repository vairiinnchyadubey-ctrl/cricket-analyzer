"""External feature scraper:
  1. Manual venue attribute table (lat, lon, altitude_m, pitch_type, boundary_m)
  2. Open-Meteo historical weather per (venue, date) -> temp, humidity, wind, precip
  3. Derived dew flag (humid + month + likely night match)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

# Manual: venue -> attributes. Ground dimensions are approximate (avg straight + square)
# Pitch type: 'spin' (red soil, low bounce, turning), 'pace' (grass, bounce, seam),
# 'flat' (Bombay clay, batting paradise)
VENUES = {
    "Wankhede Stadium, Mumbai":                                                                {"lat": 18.939, "lon": 72.825, "alt": 14,   "pitch": "flat", "boundary_m": 68},
    "MA Chidambaram Stadium, Chepauk, Chennai":                                                {"lat": 13.063, "lon": 80.279, "alt": 6,    "pitch": "spin", "boundary_m": 70},
    "M Chinnaswamy Stadium, Bengaluru":                                                        {"lat": 12.978, "lon": 77.599, "alt": 920,  "pitch": "flat", "boundary_m": 65},
    "M.Chinnaswamy Stadium":                                                                    {"lat": 12.978, "lon": 77.599, "alt": 920,  "pitch": "flat", "boundary_m": 65},
    "Eden Gardens, Kolkata":                                                                   {"lat": 22.564, "lon": 88.343, "alt": 9,    "pitch": "pace", "boundary_m": 72},
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad":                                    {"lat": 17.405, "lon": 78.553, "alt": 542,  "pitch": "flat", "boundary_m": 70},
    "Rajiv Gandhi International Stadium":                                                       {"lat": 17.405, "lon": 78.553, "alt": 542,  "pitch": "flat", "boundary_m": 70},
    "Arun Jaitley Stadium, Delhi":                                                             {"lat": 28.638, "lon": 77.243, "alt": 216,  "pitch": "flat", "boundary_m": 65},
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh":                 {"lat": 30.694, "lon": 76.808, "alt": 348,  "pitch": "flat", "boundary_m": 71},
    "Sawai Mansingh Stadium, Jaipur":                                                          {"lat": 26.892, "lon": 75.806, "alt": 432,  "pitch": "flat", "boundary_m": 67},
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow":                   {"lat": 26.773, "lon": 81.005, "alt": 123,  "pitch": "spin", "boundary_m": 68},
    "Narendra Modi Stadium, Ahmedabad":                                                        {"lat": 23.092, "lon": 72.597, "alt": 53,   "pitch": "flat", "boundary_m": 80},
    "Barsapara Cricket Stadium, Guwahati":                                                     {"lat": 26.137, "lon": 91.770, "alt": 49,   "pitch": "pace", "boundary_m": 67},
    "Himachal Pradesh Cricket Association Stadium, Dharamsala":                                {"lat": 32.190, "lon": 76.260, "alt": 1457, "pitch": "pace", "boundary_m": 68},
    "Punjab Cricket Association IS Bindra Stadium, Mohali":                                    {"lat": 30.691, "lon": 76.732, "alt": 312,  "pitch": "pace", "boundary_m": 70},
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh":                        {"lat": 30.691, "lon": 76.732, "alt": 312,  "pitch": "pace", "boundary_m": 70},
    "Maharashtra Cricket Association Stadium, Pune":                                           {"lat": 18.671, "lon": 73.917, "alt": 559,  "pitch": "flat", "boundary_m": 70},
    "Dr DY Patil Sports Academy, Mumbai":                                                       {"lat": 19.043, "lon": 73.020, "alt": 11,   "pitch": "flat", "boundary_m": 70},
    "Brabourne Stadium, Mumbai":                                                                {"lat": 18.934, "lon": 72.825, "alt": 14,   "pitch": "flat", "boundary_m": 68},
}


def venue_attrs(venue: str) -> dict:
    return VENUES.get(venue, {"lat": None, "lon": None, "alt": 200, "pitch": "flat", "boundary_m": 70})


def fetch_weather(lat: float, lon: float, date: str) -> dict:
    if lat is None:
        return {"temp_c": 28, "humidity": 60, "wind_kmh": 12, "precip_mm": 0}
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
           "&daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum"
           "&timezone=Asia%2FKolkata")
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json().get("daily", {})
        return {
            "temp_c":   float(data.get("temperature_2m_max",       [28])[0]) if data else 28.0,
            "humidity": float(data.get("relative_humidity_2m_mean",[60])[0]) if data else 60.0,
            "wind_kmh": float(data.get("wind_speed_10m_max",       [12])[0]) if data else 12.0,
            "precip_mm":float(data.get("precipitation_sum",        [0])[0])  if data else 0.0,
        }
    except Exception:
        return {"temp_c": 28, "humidity": 60, "wind_kmh": 12, "precip_mm": 0}


def main() -> None:
    matches = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    matches = matches[matches["winner"].notna()].sort_values("date").reset_index(drop=True)

    # Cache by (venue, date) so we don't re-hit the API
    cache_path = PROC / "weather_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    rows = []
    pitch_to_int = {"spin": 0, "flat": 1, "pace": 2}

    for _, m in tqdm(matches.iterrows(), total=len(matches), desc="external features"):
        v = venue_attrs(m["venue"])
        date_s = m["date"].strftime("%Y-%m-%d")
        key = f"{v['lat']},{v['lon']},{date_s}"
        if key in cache:
            w = cache[key]
        else:
            w = fetch_weather(v["lat"], v["lon"], date_s)
            cache[key] = w
            time.sleep(0.05)  # be polite to the API

        # Dew score: humid + summer night => higher
        month = m["date"].month
        is_summer = int(month in (3, 4, 5))
        dew_score = max(0.0, (w["humidity"] - 50) / 50) * (1.0 if is_summer else 0.5)

        rows.append({
            "match_id": m["match_id"],
            "alt_m": v["alt"],
            "boundary_m": v["boundary_m"],
            "pitch_type": pitch_to_int[v["pitch"]],
            "temp_c": w["temp_c"],
            "humidity": w["humidity"],
            "wind_kmh": w["wind_kmh"],
            "precip_mm": w["precip_mm"],
            "dew_score": dew_score,
            "is_summer": is_summer,
        })

    cache_path.write_text(json.dumps(cache))
    df = pd.DataFrame(rows)
    out = PROC / "external_features.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} external feature rows -> {out}")
    print(f"Weather cache: {len(cache)} entries -> {cache_path}")


if __name__ == "__main__":
    main()
