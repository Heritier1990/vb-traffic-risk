# HMMS signals dataset excluded — no Virginia Beach signals present in state VDOT file.
# VB signals are city-maintained and not included in this dataset.

import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"

CRASH_FILE  = RAW / "Police_Traffic_Crash_Reports.csv"
VOLUME_FILE = RAW / "VDOT_Bidirectional_Traffic_Volume_2024.csv"

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def basic_info(df: pd.DataFrame, name: str) -> None:
    section(f"{name} — basic info")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")
    print("\nNull counts per column (non-zero only):")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("  (no nulls)")
    else:
        print(nulls.to_string())
    print("\n5 sample rows:")
    print(df.head(5).to_string())


# ── Crash ──────────────────────────────────────────────────────────────────
crash = pd.read_csv(CRASH_FILE, low_memory=False)
basic_info(crash, "Police_Traffic_Crash_Reports")

section("Crash — year distribution from Accident_Date")
crash["_year"] = pd.to_datetime(crash["Accident_Date"], errors="coerce").dt.year
print(crash["_year"].value_counts().sort_index().to_string())

section("Crash — Virginia Beach confirmation (City column)")
city_counts = crash["City"].str.strip().str.upper().value_counts()
print(city_counts.to_string())
all_vb = (crash["City"].str.strip().str.upper() == "VIRGINIA BEACH").all()
print(f"\nAll rows from Virginia Beach: {all_vb}")

# ── Volume ─────────────────────────────────────────────────────────────────
volume = pd.read_csv(VOLUME_FILE, low_memory=False)
basic_info(volume, "VDOT_Bidirectional_Traffic_Volume_2024")

section("Volume — unique FROM_JURISDICTION values")
print(volume["FROM_JURISDICTION"].value_counts().to_string())

section("Volume — unique TO_JURISDICTION values")
print(volume["TO_JURISDICTION"].value_counts().to_string())

