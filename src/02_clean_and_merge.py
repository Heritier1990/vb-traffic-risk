# HMMS signals dataset excluded — no Virginia Beach signals present in state VDOT file.
# VB signals are city-maintained and not included in this dataset.

import re
import subprocess
import sys

import pandas as pd
from pathlib import Path

# Ensure rapidfuzz is available
try:
    from rapidfuzz import process as rf_process
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz", "-q"])
    from rapidfuzz import process as rf_process

RAW  = Path(__file__).parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


# Abbreviation map applied with word boundaries, longest forms first
_ABBREV = [
    (r"\bBOULEVARD\b", "BL"),
    (r"\bBLVD\b",      "BL"),
    (r"\bPARKWAY\b",   "PW"),
    (r"\bPKWY\b",      "PW"),
    (r"\bROAD\b",      "RD"),
    (r"\bSTREET\b",    "ST"),
    (r"\bAVENUE\b",    "AV"),
    (r"\bDRIVE\b",     "DR"),
    (r"\bHIGHWAY\b",   "HWY"),
    (r"\bLANE\b",      "LN"),
    (r"\bCOURT\b",     "CT"),
    (r"\bPLACE\b",     "PL"),
]
_ABBREV_RE = [(re.compile(p), r) for p, r in _ABBREV]
_PUNCT_RE  = re.compile(r"[^\w\s]")
_SPACE_RE  = re.compile(r"\s+")


def normalize_road_key(s: str) -> str:
    for pattern, replacement in _ABBREV_RE:
        s = pattern.sub(replacement, s)
    s = _PUNCT_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


# ── 1. Crash data ──────────────────────────────────────────────────────────
section("Crash data — load and filter")

crash_raw = pd.read_csv(RAW / "Police_Traffic_Crash_Reports.csv", low_memory=False)
print(f"Shape before:                          {crash_raw.shape}")

crash_raw["Accident_Date"] = pd.to_datetime(crash_raw["Accident_Date"], errors="coerce")
crash = crash_raw[crash_raw["Accident_Date"].dt.year == 2023].copy()
print(f"Shape after year == 2023 filter:       {crash.shape}")

# Hour from Accident_Time: zero-pad to 4 digits, take first 2 chars
crash["Accident_Time"] = crash["Accident_Time"].astype(str).str.zfill(4)
crash["hour"] = crash["Accident_Time"].str[:2].astype(int)

# Clean Main_Street
crash["Main_Street"] = crash["Main_Street"].astype(str).str.strip().str.upper()
before_blank = len(crash)
crash = crash[~crash["Main_Street"].isin(["", "NAN"])].copy()
print(f"Rows dropped (blank/null Main_Street): {before_blank - len(crash)}")
print(f"Shape after blank Main_Street drop:    {crash.shape}")

# Drop nearly-empty work-zone columns
crash.drop(
    columns=["Work_Zone_Location", "Work_Zone_Type", "Workers_Present_in_Work_Zone"],
    errors="ignore",
    inplace=True,
)

KEEP = [
    "hour", "Day_of_Week", "Main_Street", "Nearest_Street",
    "At_Intersection", "Light_Conditions", "Weather_Condition",
    "Roadway_Surface_Condition", "Traffic_Control_Device",
    "Intersection_Type", "Type_of_Collision", "Number_of_Vehicles_Involved",
    "School_Zone", "Work_Zone_Related", "Zone_ID",
]
crash = crash[[c for c in KEEP if c in crash.columns]].copy()
crash["road_key"]      = crash["Main_Street"]
crash["road_key_norm"] = crash["road_key"].apply(normalize_road_key)
print(f"Final crash shape:                     {crash.shape}")


# ── 2. Volume data ─────────────────────────────────────────────────────────
section("Volume data — load and filter")

vol_raw = pd.read_csv(RAW / "VDOT_Bidirectional_Traffic_Volume_2024.csv", low_memory=False)
print(f"Shape before:                                  {vol_raw.shape}")

vb_mask = (
    vol_raw["FROM_JURISDICTION"].str.contains("Virginia Beach", case=False, na=False)
    | vol_raw["TO_JURISDICTION"].str.contains("Virginia Beach", case=False, na=False)
)
vol = vol_raw[vb_mask].copy()
print(f"Shape after Virginia Beach jurisdiction filter: {vol.shape}")

# road_key: text before first '(', stripped and uppercased
vol["road_key"] = (
    vol["ROUTE_COMMON_NAME"]
    .str.split("(").str[0]
    .str.strip()
    .str.upper()
)
vol["road_key_norm"] = vol["road_key"].apply(normalize_road_key)

vol = vol[["road_key", "road_key_norm", "ADT", "AAWDT"]].copy()
vol = vol.dropna(subset=["ADT"])

# Keep highest ADT per normalized key
vol = vol.sort_values("ADT", ascending=False).drop_duplicates(subset=["road_key_norm"])
print(f"Shape after deduplication (highest ADT):       {vol.shape}")
print("\n10 sample road_key_norm values:")
print(vol["road_key_norm"].head(10).to_string())


# ── 3. Merge — crash + volume ─────────────────────────────────────────────
section("Merge — exact join on normalized road_key")

vol_for_join = vol[["road_key_norm", "ADT", "AAWDT"]].copy()

merged = crash.merge(vol_for_join, on="road_key_norm", how="left")
exact_matches = merged["ADT"].notna().sum()
total_rows    = len(merged)
print(f"Total crash rows:          {total_rows:,}")
print(f"Matched by exact join:     {exact_matches:,}  ({exact_matches/total_rows:.1%})")


# ── 4. Fuzzy join for rows still missing ADT ──────────────────────────────
section("Merge — fuzzy join for unmatched rows (threshold = 80)")

THRESHOLD = 80
vol_keys   = vol["road_key_norm"].tolist()
vol_lookup = vol.set_index("road_key_norm")[["ADT", "AAWDT"]].to_dict("index")

unmatched_mask  = merged["ADT"].isna()
unique_unmatched = merged.loc[unmatched_mask, "road_key_norm"].unique()
print(f"Unique unmatched road keys to fuzzy-match: {len(unique_unmatched)}")

fuzzy_map: dict[str, tuple] = {}
for key in unique_unmatched:
    result = rf_process.extractOne(key, vol_keys, score_cutoff=THRESHOLD)
    if result:
        matched_key = result[0]
        fuzzy_map[key] = (
            vol_lookup[matched_key]["ADT"],
            vol_lookup[matched_key]["AAWDT"],
            matched_key,
        )

fuzzy_adt  = merged.loc[unmatched_mask, "road_key_norm"].map(
    {k: v[0] for k, v in fuzzy_map.items()}
)
fuzzy_aawdt = merged.loc[unmatched_mask, "road_key_norm"].map(
    {k: v[1] for k, v in fuzzy_map.items()}
)

merged.loc[unmatched_mask, "ADT"]   = fuzzy_adt.values
merged.loc[unmatched_mask, "AAWDT"] = fuzzy_aawdt.values
merged.loc[unmatched_mask, "match_type"] = merged.loc[
    unmatched_mask, "road_key_norm"
].map({k: "fuzzy" for k in fuzzy_map}).values

# Rows that got exact matches
still_unmatched_mask = merged["ADT"].isna()
merged.loc[~still_unmatched_mask & merged["match_type"].isna(), "match_type"] = "exact"

fuzzy_matches = merged.loc[unmatched_mask & merged["ADT"].notna()].shape[0]
still_unmatched = still_unmatched_mask.sum()

print(f"Matched by fuzzy join:     {fuzzy_matches:,}  ({fuzzy_matches/total_rows:.1%})")
print(f"Still unmatched:           {still_unmatched:,}  ({still_unmatched/total_rows:.1%})")


# ── 5. Median fill for remaining unmatched rows ───────────────────────────
section("Median fill for remaining unmatched rows")

median_adt = vol["ADT"].median()
print(f"Median ADT from Virginia Beach volume data: {median_adt:,.0f}")

merged.loc[still_unmatched_mask, "ADT"]        = median_adt
merged.loc[still_unmatched_mask, "AAWDT"]      = vol["AAWDT"].median()
merged.loc[still_unmatched_mask, "match_type"] = "median_fill"

print("\nMatch type breakdown:")
print(merged["match_type"].value_counts().to_string())


# ── 6. Final output ───────────────────────────────────────────────────────
section("Final dataset")

# Drop the normalized key column — it was internal scaffolding
merged.drop(columns=["road_key_norm"], inplace=True)

print(f"Shape: {merged.shape}")
print(f"Columns: {list(merged.columns)}")
print("\nFirst 5 rows:")
print(merged.head(5).to_string())

out_path = PROC / "merged_2023.csv"
merged.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
