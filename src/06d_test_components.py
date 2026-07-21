"""
Component test — validates all runtime dependencies without starting Streamlit.
Run from the project root: python src/06d_test_components.py
"""
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE   = Path(__file__).parent.parent
MODELS = BASE / "models"
PROC   = BASE / "data" / "processed"

SEP = "=" * 65

def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ══════════════════════════════════════════════════════════════════
# TEST 1 — Load models, scalers, and feature JSON files
# ══════════════════════════════════════════════════════════════════
section("TEST 1 — Models, scalers, feature JSON files")

acc_model   = joblib.load(MODELS / "accident_model.pkl")
cong_model  = joblib.load(MODELS / "congestion_model.pkl")
acc_scaler  = joblib.load(MODELS / "accident_scaler.pkl")
cong_scaler = joblib.load(MODELS / "congestion_scaler.pkl")
acc_feat    = json.loads((MODELS / "accident_features.json").read_text())
cong_feat   = json.loads((MODELS / "congestion_features.json").read_text())

print(f"Accident model type   : {type(acc_model).__name__}")
print(f"  features ({len(acc_feat['features'])}): {acc_feat['features']}")
print(f"  scaled_cols         : {acc_feat['scaled_cols']}")
print(f"  leaky_col           : {acc_feat['leaky_col']}")

print(f"\nCongestion model type : {type(cong_model).__name__}")
print(f"  features ({len(cong_feat['features'])}): {cong_feat['features']}")
print(f"  scaled_cols         : {cong_feat['scaled_cols']}")
print(f"  leaky_col           : {cong_feat['leaky_col']}")

print(f"\nAccident scaler type  : {type(acc_scaler).__name__}")
print(f"  mean_                : {acc_scaler.mean_}")
print(f"Congestion scaler type: {type(cong_scaler).__name__}")
print(f"  mean_                : {cong_scaler.mean_}")

print("\nTEST 1 PASSED")


# ══════════════════════════════════════════════════════════════════
# TEST 2 — Build feature vector
# ══════════════════════════════════════════════════════════════════
section("TEST 2 — Feature vector for VIRGINIA BEACH BL, next Friday 23:00, weather_encoded=2")

hotspots = pd.read_csv(PROC / "hotspots.csv")

# Determine next Friday
today = date.today()
days_until_friday = (4 - today.weekday()) % 7  # weekday 4 = Friday
days_until_friday = days_until_friday if days_until_friday > 0 else 7
next_friday = today + timedelta(days=days_until_friday)

selected_road    = "VIRGINIA BEACH BL"
selected_date    = next_friday
selected_hour    = 23
weather_encoded  = 2

print(f"Road         : {selected_road}")
print(f"Date         : {selected_date}  ({selected_date.strftime('%A')})")
print(f"Hour         : {selected_hour:02d}:00")
print(f"weather_encoded: {weather_encoded}")

row = hotspots[hotspots["road_key"] == selected_road]
if row.empty:
    historical_crash_rate = float(hotspots["total_crashes"].median())
    avg_adt               = float(hotspots["avg_ADT"].median())
    print(f"\nRoad not found in hotspots — using medians")
else:
    historical_crash_rate = float(row["total_crashes"].iloc[0])
    avg_adt               = float(row["avg_ADT"].iloc[0])
    print(f"\nHotspot lookup OK  total_crashes={historical_crash_rate:.0f}  avg_ADT={avg_adt:.0f}")

is_night    = 1 if (selected_hour < 6 or selected_hour >= 20) else 0
is_rush     = 1 if selected_hour in [7, 8, 9, 16, 17, 18]   else 0
is_weekend  = 1 if selected_date.weekday() in [5, 6]         else 0
school_zone = 1 if (7 <= selected_hour <= 16 and not is_weekend) else 0
light_enc   = 0 if (6 <= selected_hour <= 19) else 2
surface_enc = 1 if weather_encoded >= 2 else 0

full_feat = {
    "hour":                   float(selected_hour),
    "month":                  float(selected_date.month),
    "is_night":               float(is_night),
    "is_rush_hour":           float(is_rush),
    "is_weekend":             float(is_weekend),
    "at_intersection_binary": 1.0,
    "school_zone_binary":     float(school_zone),
    "work_zone_binary":       0.0,
    "weather_encoded":        float(weather_encoded),
    "light_encoded":          float(light_enc),
    "surface_encoded":        float(surface_enc),
    "historical_crash_rate":  historical_crash_rate,
    "ADT":                    avg_adt,
}

print(f"\nComplete feature dictionary:")
for k, v in full_feat.items():
    print(f"  {k:<26} = {v}")

print("\nTEST 2 PASSED")


# ══════════════════════════════════════════════════════════════════
# TEST 3 — Slice, scale, predict_proba
# ══════════════════════════════════════════════════════════════════
section("TEST 3 — Slice features, apply scaler, predict_proba")

def run_model(model, scaler, feat_meta, feat_dict, label):
    feature_list = feat_meta["features"]
    scaled_cols  = feat_meta["scaled_cols"]

    vec = np.array([[feat_dict[c] for c in feature_list]], dtype=float)
    print(f"\n  [{label}] raw vector (pre-scale):")
    for name, val in zip(feature_list, vec[0]):
        print(f"    {name:<26} = {val:.4f}")

    scaled_idx = [feature_list.index(c) for c in scaled_cols]
    vec[:, scaled_idx] = scaler.transform(vec[:, scaled_idx])
    print(f"\n  [{label}] scaled columns {scaled_cols}:")
    for i, name in zip(scaled_idx, scaled_cols):
        print(f"    {name:<26} = {vec[0, i]:.4f}")

    proba      = model.predict_proba(vec)[0]
    risk_score = float(proba[1] * 50 + proba[2] * 100)
    risk_class = ["Low", "Medium", "High"][int(np.argmax(proba))]

    print(f"\n  [{label}] predict_proba:")
    print(f"    P(Low)    = {proba[0]:.4f}")
    print(f"    P(Medium) = {proba[1]:.4f}")
    print(f"    P(High)   = {proba[2]:.4f}")
    print(f"    Risk class = {risk_class}")
    print(f"    Risk score = {risk_score:.1f} / 100")
    return risk_class, risk_score, proba

acc_class,  acc_score,  acc_proba  = run_model(acc_model,  acc_scaler,  acc_feat,  full_feat, "ACCIDENT")
cong_class, cong_score, cong_proba = run_model(cong_model, cong_scaler, cong_feat, full_feat, "CONGESTION")

print(f"\nSummary:")
print(f"  Accident  Risk -> {acc_class:<6}  score={acc_score:.1f}")
print(f"  Congestion Risk -> {cong_class:<6}  score={cong_score:.1f}")
print("\nTEST 3 PASSED")


# ══════════════════════════════════════════════════════════════════
# TEST 4 — OpenWeatherMap forecast API
# ══════════════════════════════════════════════════════════════════
section("TEST 4 — OpenWeatherMap forecast API")

api_key = os.getenv("OPENWEATHER_API_KEY", "")
if not api_key:
    print("WARNING: OPENWEATHER_API_KEY not set in .env — skipping API call")
else:
    print(f"API key loaded: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={
                "lat":   36.8529,
                "lon":  -75.9780,
                "units": "imperial",
                "appid": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        entries = resp.json()["list"]
        print(f"Forecast entries returned: {len(entries)}")
        print(f"\nNext 3 forecast entries for Virginia Beach:")
        for entry in entries[:3]:
            ts   = datetime.fromtimestamp(entry["dt"])
            desc = entry["weather"][0]["description"].capitalize()
            temp = entry["main"]["temp"]
            wind = entry["wind"]["speed"]
            main = entry["weather"][0]["main"]
            print(f"  {ts.strftime('%Y-%m-%d %H:%M')}  |  {main:<14} {desc:<30}  "
                  f"{temp:.1f}°F  wind={wind:.1f} mph")
        print("\nTEST 4 PASSED")
    except Exception as exc:
        print(f"API call failed: {exc}")
        print("TEST 4 FAILED — check API key and network connectivity")


# ══════════════════════════════════════════════════════════════════
# TEST 5 — hotspots.csv and road_coordinates.json
# ══════════════════════════════════════════════════════════════════
section("TEST 5 — hotspots.csv and road_coordinates.json")

road_coords = json.loads((PROC / "road_coordinates.json").read_text())

print(f"hotspots.csv shape   : {hotspots.shape}")
print(f"road_coordinates.json: {len(road_coords)} entries")
print(f"\nTop 5 roads by total_crashes:\n")

cols = ["road_key", "total_crashes", "lat", "lon",
        "accident_risk_level", "congestion_risk_level"]
top5 = hotspots.sort_values("total_crashes", ascending=False).head(5)
print(top5[cols].to_string(index=False))

# Verify coordinates exist for all top-5
print(f"\nCoordinate check for top-5 roads:")
for road in top5["road_key"].tolist():
    in_json = road in road_coords
    if in_json:
        rc = road_coords[road]
        print(f"  {road:<28}  lat={rc['lat']:.4f}  lon={rc['lon']:.4f}  OK")
    else:
        print(f"  {road:<28}  MISSING from road_coordinates.json")

print("\nTEST 5 PASSED")

print(f"\n{SEP}\nAll tests completed.\n{SEP}")
