import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"

SEP = "=" * 65

def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


# ── 1. Load ────────────────────────────────────────────────────────────────
section("Load")

# Accident_Date was not kept in the merge pipeline — re-read raw to get month.
# The raw file filtered to 2023 + non-blank Main_Street produces the same 4,470
# rows in the same order as merged_2023.csv, so positional join is safe.
crash_raw = pd.read_csv(RAW / "Police_Traffic_Crash_Reports.csv", low_memory=False)
crash_raw["Accident_Date"] = pd.to_datetime(crash_raw["Accident_Date"], errors="coerce")
crash_2023 = crash_raw[crash_raw["Accident_Date"].dt.year == 2023].copy()
crash_2023["Main_Street"] = crash_2023["Main_Street"].astype(str).str.strip().str.upper()
crash_2023 = (
    crash_2023[~crash_2023["Main_Street"].isin(["", "NAN"])]
    .reset_index(drop=True)
)
print(f"Raw crash (2023, non-blank Main_Street): {len(crash_2023):,} rows")

df_raw = pd.read_csv(PROC / "merged_2023.csv").reset_index(drop=True)
print(f"Merged file:                             {len(df_raw):,} rows")

if len(df_raw) != len(crash_2023):
    raise ValueError(
        f"Row count mismatch — raw crash {len(crash_2023)} vs merged {len(df_raw)}. "
        "Re-run 02_clean_and_merge.py first."
    )

# Attach month before the median-fill filter so positional index is preserved
df_raw["month"] = crash_2023["Accident_Date"].dt.month.values

# Remove median-fill rows (304 rows with imputed ADT)
df = df_raw[df_raw["match_type"] != "median_fill"].copy().reset_index(drop=True)
print(f"After removing median_fill rows:         {len(df):,} rows")


# ── 2. Drop unused columns ─────────────────────────────────────────────────
section("Drop columns")

DROP_COLS = [
    "Main_Street",       # duplicated by road_key
    "AAWDT",             # corr 1.00 with ADT
    "Nearest_Street",    # not used as feature
    "match_type",        # pipeline artifact
    # already dropped in 02, kept here for safety:
    "Work_Zone_Location", "Work_Zone_Type", "Workers_Present_in_Work_Zone",
]
before = list(df.columns)
df.drop(columns=DROP_COLS, errors="ignore", inplace=True)
dropped = [c for c in before if c not in df.columns]
print(f"Dropped: {dropped}")
print(f"Remaining columns: {list(df.columns)}")


# ── 3. Feature engineering ─────────────────────────────────────────────────
section("Feature engineering")

# at_intersection_binary
def encode_at_intersection(val) -> float:
    if pd.isna(val):
        return np.nan
    v = str(val).strip().upper()
    if v == "YES":
        return 1.0
    if v == "NO":
        return 0.0
    if "FEET FROM" in v:
        return 0.0
    return np.nan  # Miles From and anything else

df["at_intersection_binary"] = df["At_Intersection"].apply(encode_at_intersection)

# Time-based flags
df["is_night"]     = ((df["hour"] < 6) | (df["hour"] >= 20)).astype(int)
df["is_rush_hour"] = df["hour"].isin(list(range(7, 10)) + list(range(16, 19))).astype(int)

# Day-based flag
df["is_weekend"] = (
    df["Day_of_Week"].str.strip().str.upper().isin(["SAT", "SUN"])
).astype(int)

# School / work zone
df["school_zone_binary"] = (
    df["School_Zone"].astype(str).str.strip().str.upper().str.startswith("YES")
).astype(int)

df["work_zone_binary"] = (
    df["Work_Zone_Related"].astype(str).str.strip().str.upper() == "YES"
).astype(int)

# Weather encoding
WEATHER_MAP = {
    "NO ADVERSE CONDITION (CLEAR/CLOUDY)": 0,
    "MIST":  1,
    "RAIN":  2,
    "FOG":   3,
    "SNOW":  4,
}
df["weather_encoded"] = df["Weather_Condition"].str.strip().map(WEATHER_MAP)

# Light encoding
LIGHT_MAP = {
    "DAYLIGHT":                    0,
    "DAWN":                        1,
    "DUSK":                        1,
    "DARKNESS-ROAD LIGHTED":       2,
    "DARKNESS-ROAD NOT LIGHTED":   3,
}
df["light_encoded"] = df["Light_Conditions"].str.strip().map(LIGHT_MAP)

# Surface encoding
SURFACE_MAP = {
    "DRY":   0,
    "WET":   1,
    "SNOWY": 2,
    "ICY":   3,
}
df["surface_encoded"] = df["Roadway_Surface_Condition"].str.strip().map(SURFACE_MAP)

# Historical crash rate (computed on the post-median-fill-filter universe)
crash_rate = df.groupby("road_key").size()
df["historical_crash_rate"] = df["road_key"].map(crash_rate)

print("Engineered columns added:")
new_cols = [
    "month", "at_intersection_binary", "is_night", "is_rush_hour", "is_weekend",
    "school_zone_binary", "work_zone_binary", "weather_encoded",
    "light_encoded", "surface_encoded", "historical_crash_rate",
]
for col in new_cols:
    print(f"  {col:30s}  non-null={df[col].notna().sum():,}")


# ── 4. Select final feature columns ────────────────────────────────────────
section("Select feature columns")

BASE_COLS = [
    "road_key", "hour", "month", "is_night", "is_rush_hour", "is_weekend",
    "at_intersection_binary", "school_zone_binary", "work_zone_binary",
    "weather_encoded", "light_encoded", "surface_encoded",
    "historical_crash_rate", "ADT",
]

# Optional signal columns — include only if present and not all-NaN
SIGNAL_COLS = ["signal_count", "pct_coordinated", "pct_detection"]
present_signal_cols = [c for c in SIGNAL_COLS if c in df.columns and df[c].notna().any()]
absent_signal_cols  = [c for c in SIGNAL_COLS if c not in present_signal_cols]

if absent_signal_cols:
    print(f"NOTE: {absent_signal_cols} absent — no Virginia Beach signals in HMMS dataset.")
if present_signal_cols:
    print(f"Including signal columns: {present_signal_cols}")

FEAT_COLS = BASE_COLS + present_signal_cols
feat = df[FEAT_COLS].copy()


# ── 5. Null report ─────────────────────────────────────────────────────────
section("Null report (before dropping)")

print(f"Total rows: {len(feat):,}\n")
print(f"{'Column':<30}  {'Null count':>10}  {'Null %':>8}")
print("-" * 55)
for col in feat.columns:
    n = feat[col].isna().sum()
    p = n / len(feat) * 100
    flag = "  <-- will cause row drop" if n > 0 else ""
    print(f"  {col:<28}  {n:>10,}  {p:>7.1f}%{flag}")

# Null-drop preview (exclude road_key from check — it's metadata, never null)
check_cols = [c for c in feat.columns if c != "road_key"]
would_remain = feat.dropna(subset=check_cols).shape[0]
would_drop   = len(feat) - would_remain
print(f"\nRows that would be dropped: {would_drop:,}")
print(f"Rows that would remain:     {would_remain:,}")


# ── 6. Drop nulls ──────────────────────────────────────────────────────────
section("Drop null rows")

feat = feat.dropna(subset=check_cols).reset_index(drop=True)
print(f"Shape after dropna: {feat.shape}")
assert feat[check_cols].isna().sum().sum() == 0, "Nulls remain after dropna!"
print("Null check passed — zero nulls in feature columns.")


# ── 7. Labels ──────────────────────────────────────────────────────────────
section("Label 1 — accident_risk")

def accident_risk(rate: int) -> int:
    if rate == 1:
        return 0
    if rate <= 4:
        return 1
    return 2

feat["accident_risk"] = feat["historical_crash_rate"].apply(accident_risk)

dist1 = feat["accident_risk"].value_counts().sort_index()
label_names = {0: "Low (rate==1)", 1: "Medium (rate 2-4)", 2: "High (rate>=5)"}
for lbl, cnt in dist1.items():
    print(f"  {lbl} — {label_names[lbl]:<22}  {cnt:>5,}  ({cnt/len(feat)*100:.1f}%)")


section("Label 2 — congestion_risk")

conditions = [
    feat["ADT"] < 17_000,
    (feat["ADT"] >= 17_000) & (feat["ADT"] <= 36_000),
    feat["ADT"] > 36_000,
]
feat["congestion_risk"] = np.select(conditions, [0, 1, 2])

dist2 = feat["congestion_risk"].value_counts().sort_index()
cong_names = {0: "Low  (ADT < 17,000)", 1: "Medium (17k-36k)", 2: "High (ADT > 36,000)"}
for lbl, cnt in dist2.items():
    print(f"  {int(lbl)} — {cong_names[int(lbl)]:<22}  {cnt:>5,}  ({cnt/len(feat)*100:.1f}%)")


# ── 8. Save ────────────────────────────────────────────────────────────────
section("Save")

ACC_COLS  = FEAT_COLS + ["accident_risk"]
CONG_COLS = FEAT_COLS + ["congestion_risk"]

acc_df  = feat[ACC_COLS]
cong_df = feat[CONG_COLS]

acc_path  = PROC / "accident_dataset.csv"
cong_path = PROC / "congestion_dataset.csv"

acc_df.to_csv(acc_path,  index=False)
cong_df.to_csv(cong_path, index=False)

for path, out_df, name in [
    (acc_path,  acc_df,  "accident_dataset.csv"),
    (cong_path, cong_df, "congestion_dataset.csv"),
]:
    nulls = out_df.drop(columns=["road_key"]).isna().sum().sum()
    print(f"  {name:<30}  shape={out_df.shape}  nulls={nulls}")

print(f"\nSaved to {PROC}")
