"""
06a_geocode_roads.py
Geocode all unique roads from merged_2023.csv using Nominatim.

- Normalises road names the same way app.py does (strip / upper / drop leading digits).
- Keeps existing valid entries from road_coordinates.json; only geocodes missing ones.
- Accepts coordinates only inside the Virginia Beach bounding box.
- Retries once with expanded abbreviations if the first attempt fails.
- Saves incrementally every 10 roads.
- Prints progress every 10 roads and a final summary.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# -- Auto-install dependencies --------------------------------------------------
for pkg, mod in [("pandas", "pandas"), ("geopy", "geopy")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import pandas as pd
from geopy.geocoders import Nominatim

# -- Paths ----------------------------------------------------------------------
BASE        = Path(__file__).parent.parent
PROC        = BASE / "data" / "processed"
COORDS_PATH = PROC / "road_coordinates.json"

# -- Virginia Beach bounding box ------------------------------------------------
VB_LAT_MIN, VB_LAT_MAX =  36.65,  36.95
VB_LON_MIN, VB_LON_MAX = -76.10, -75.90

# -- Abbreviation expansions used on retry --------------------------------------
_ABBREV = [
    (r"\bBL\b",   "Boulevard"),
    (r"\bBLVD\b", "Boulevard"),
    (r"\bPW\b",   "Parkway"),
    (r"\bPKWY\b", "Parkway"),
    (r"\bRD\b",   "Road"),
    (r"\bDR\b",   "Drive"),
    (r"\bST\b",   "Street"),
    (r"\bAVE\b",  "Avenue"),
    (r"\bCT\b",   "Court"),
    (r"\bLN\b",   "Lane"),
    (r"\bHWY\b",  "Highway"),
]

SEP = "=" * 65


# -- Helpers --------------------------------------------------------------------
def normalize(name: str) -> str:
    """Match the normalization used in app.py load_all_roads()."""
    name = str(name).strip().upper()
    return re.sub(r"^\d+\s+", "", name)


def expand_abbrevs(name: str) -> str:
    """Remove leading digits then expand common road-name abbreviations."""
    name = re.sub(r"^\d+\s+", "", name)
    for pattern, replacement in _ABBREV:
        name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
    return name.strip()


def in_vb(lat: float, lon: float) -> bool:
    return VB_LAT_MIN <= lat <= VB_LAT_MAX and VB_LON_MIN <= lon <= VB_LON_MAX


def geocode_one(geo, query: str) -> dict | None:
    """Return coord dict if result is within VB bounds, else None."""
    try:
        loc = geo.geocode(query, timeout=10)
        if loc and in_vb(loc.latitude, loc.longitude):
            return {"lat": loc.latitude, "lon": loc.longitude,
                    "display_name": loc.address}
    except Exception as exc:
        print(f"    geocode error: {exc}")
    return None


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


# -- Load all unique roads from CSV --------------------------------------------
print(SEP)
print("Loading roads from merged_2023.csv ...")
df       = pd.read_csv(PROC / "merged_2023.csv", usecols=["road_key"])
all_roads = (
    df["road_key"]
    .dropna()
    .map(normalize)
    .drop_duplicates()
    .sort_values()
    .tolist()
)
print(f"  {len(all_roads)} unique roads after normalisation and deduplication")

# -- Load existing valid entries -----------------------------------------------
existing: dict = {}
if COORDS_PATH.exists():
    raw = json.loads(COORDS_PATH.read_text())
    for road, data in raw.items():
        if in_vb(data.get("lat", 0.0), data.get("lon", 0.0)):
            existing[road] = data
    print(f"  {len(existing)} existing valid entries loaded from {COORDS_PATH.name}")
else:
    print("  No existing road_coordinates.json found - starting fresh")

to_geocode = [r for r in all_roads if r not in existing]
already    = len(all_roads) - len(to_geocode)
print(f"  {already} already geocoded - {len(to_geocode)} remaining")
print(SEP)

if not to_geocode:
    print("Nothing to do. Exiting.")
    raise SystemExit(0)

# -- Geocode -------------------------------------------------------------------
geo    = Nominatim(user_agent="vb_crash_risk_geocoder_v2")
coords = dict(existing)  # build on top of existing valid entries

succeeded = 0
failed    = 0
total     = len(to_geocode)

for idx, road in enumerate(to_geocode, start=1):

    # Primary attempt
    result = geocode_one(geo, f"{road}, Virginia Beach, VA, USA")
    time.sleep(1.1)

    # Retry with expanded abbreviations
    if result is None:
        expanded = expand_abbrevs(road)
        if expanded != road:
            result = geocode_one(geo, f"{expanded}, Virginia Beach, VA, USA")
            time.sleep(1.1)

    if result:
        coords[road] = result
        succeeded += 1
        print(f"  OK   [{idx:>4}/{total}] {road:<38} "
              f"{result['lat']:.4f}, {result['lon']:.4f}")
    else:
        failed += 1
        print(f"  FAIL [{idx:>4}/{total}] {road}")

    # Incremental save + progress every 10 roads
    if idx % 10 == 0 or idx == total:
        save(COORDS_PATH, coords)
        remaining = total - idx
        print(f"\n  -- checkpoint {idx}/{total} | "
              f"succeeded={succeeded}  failed={failed}  remaining={remaining} --\n")

# -- Final summary -------------------------------------------------------------
print(SEP)
print("Geocoding complete")
print(f"  Roads attempted      : {total}")
print(f"  Successfully geocoded: {succeeded}")
print(f"  Failed               : {failed}")
print(f"  Total entries in file: {len(coords)}  (includes {already} pre-existing)")
print(f"  Saved to             : {COORDS_PATH}")
print(SEP)
