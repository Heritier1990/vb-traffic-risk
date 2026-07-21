import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
PROC = BASE / "data" / "processed"

# ── 1. Load ────────────────────────────────────────────────────────────────
df = pd.read_csv(PROC / "merged_2023.csv")
print(f"Loaded merged_2023.csv: {df.shape}")

if "match_type" in df.columns:
    before = len(df)
    df = df[df["match_type"] != "median_fill"].copy()
    print(f"Dropped {before - len(df)} median_fill rows — {len(df)} remain")

coords_raw = json.loads((PROC / "road_coordinates.json").read_text())
print(f"Loaded road_coordinates.json: {len(coords_raw)} entries")

# ── 2. Normalize road_key to match the dropdown pipeline ───────────────────
# Strip, uppercase, remove leading block numbers ("200 14TH ST" -> "14TH ST")
df["road_key"] = (
    df["road_key"]
    .fillna("")
    .str.strip()
    .str.upper()
    .str.replace(r"^\d+\s+", "", regex=True)
)
before_norm = df["road_key"].nunique()
print(f"Unique road_key values after normalization: {before_norm}  "
      f"(was {pd.read_csv(PROC / 'merged_2023.csv', usecols=['road_key'])['road_key'].nunique()} raw)")

# ── 3. Per-road aggregation ────────────────────────────────────────────────
df["_is_night"]   = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
df["_is_rush"]    = df["hour"].isin(list(range(7, 10)) + list(range(16, 19))).astype(int)
df["_is_weekend"] = df["Day_of_Week"].str.strip().str.upper().isin(["SAT", "SUN"]).astype(int)

agg = df.groupby("road_key").agg(
    total_crashes   = ("road_key",    "count"),
    night_crashes   = ("_is_night",   "sum"),
    rush_crashes    = ("_is_rush",    "sum"),
    weekend_crashes = ("_is_weekend", "sum"),
    avg_ADT         = ("ADT",         "mean"),
).reset_index()

print(f"\nRoads aggregated: {len(agg)}")

# ── 4. Risk classification ─────────────────────────────────────────────────
def accident_risk(n):
    if n == 1:   return "Low"
    if n <= 4:   return "Medium"
    return "High"

def congestion_risk(adt):
    if adt < 17_000:  return "Low"
    if adt <= 36_000: return "Medium"
    return "High"

agg["accident_risk_level"]   = agg["total_crashes"].apply(accident_risk)
agg["congestion_risk_level"] = agg["avg_ADT"].apply(congestion_risk)

# ── 5. Join coordinates ────────────────────────────────────────────────────
coord_df = pd.DataFrame([
    {"road_key": k, "lat": v["lat"], "lon": v["lon"]}
    for k, v in coords_raw.items()
])

# Left join — all roads kept; unmatched get lat=NaN, lon=NaN
hotspots = agg.merge(coord_df, on="road_key", how="left")

# ── 6. Round numeric columns ───────────────────────────────────────────────
hotspots["avg_ADT"] = hotspots["avg_ADT"].fillna(0).round(0).astype(int)

# ── 7. Save ────────────────────────────────────────────────────────────────
out_path = PROC / "hotspots.csv"
hotspots.to_csv(out_path, index=False)

# ── 8. Summary ────────────────────────────────────────────────────────────
with_coords  = hotspots["lat"].notna().sum()
without_coords = hotspots["lat"].isna().sum()
risk_dist    = hotspots["accident_risk_level"].value_counts().to_dict()

print(f"\nSaved {out_path}")
print(f"  Total roads in hotspots.csv : {len(hotspots)}")
print(f"  Roads with coordinates      : {with_coords}")
print(f"  Roads without coordinates   : {without_coords}")
print(f"  Accident risk distribution  :")
for level in ["Low", "Medium", "High"]:
    n = risk_dist.get(level, 0)
    print(f"    {level:<8}: {n:>4}  ({n/len(hotspots)*100:.1f}%)")

print(f"\nTop 10 roads by crash count:\n")
cols = ["road_key", "total_crashes", "lat", "lon",
        "accident_risk_level", "congestion_risk_level"]
top10 = hotspots.sort_values("total_crashes", ascending=False).head(10)
print(top10[cols].to_string(index=False))
