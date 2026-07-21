"""
Virginia Beach Traffic and Accident Risk Intelligence
Streamlit application
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Groq (auto-install if missing) ────────────────────────────────────────────
try:
    from groq import Groq as _Groq
    _GROQ_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "groq", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from groq import Groq as _Groq
        _GROQ_AVAILABLE = True
    except Exception:
        _GROQ_AVAILABLE = False

# ── Third-party imports ────────────────────────────────────────────────────────
import folium
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# ── Environment — works locally (.env) and on Streamlit Cloud (st.secrets) ────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

try:
    OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
except Exception:
    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Design system constants ────────────────────────────────────────────────────
DS_BG_DARK    = "#01131F"
DS_OCEAN_BLUE = "#00A8CC"
DS_CARD_BG    = "rgba(0,0,0,0.55)"
DS_TEXT_WHITE = "#FFFFFF"
DS_TEXT_SKY   = "#ADE8F4"
DS_HIGHLIGHT  = "#FFD700"
DS_SUCCESS    = "#0ACF83"
DS_WARNING    = "#F4A261"
DS_DANGER     = "#E63946"

DS_FONT_TITLE    = "52px"
DS_FONT_HEADER   = "28px"
DS_FONT_SUBTITLE = "20px"
DS_FONT_BODY     = "17px"
DS_FONT_SMALL    = "14px"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
ASSETS = BASE / "assets"
PROC   = BASE / "data" / "processed"
MODELS = BASE / "models"

# ── Page config (must be first Streamlit call) ─────────────────────────────────
_icon_path = ASSETS / "traffic_icon.png"
st.set_page_config(
    page_title="Virginia Beach Traffic and Accident Risk Intelligence",
    page_icon=str(_icon_path) if _icon_path.exists() else "🚦",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "screen" not in st.session_state:
    st.session_state.screen = "landing"
if "results" not in st.session_state:
    st.session_state.results = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    acc_model   = joblib.load(MODELS / "accident_model.pkl")
    cong_model  = joblib.load(MODELS / "congestion_model.pkl")
    acc_scaler  = joblib.load(MODELS / "accident_scaler.pkl")
    cong_scaler = joblib.load(MODELS / "congestion_scaler.pkl")
    acc_feat    = json.loads((MODELS / "accident_features.json").read_text())
    cong_feat   = json.loads((MODELS / "congestion_features.json").read_text())
    hotspots    = pd.read_csv(PROC / "hotspots.csv")
    road_coords = json.loads((PROC / "road_coordinates.json").read_text())
    return (
        acc_model, cong_model,
        acc_scaler, cong_scaler,
        acc_feat, cong_feat,
        hotspots, road_coords,
    )


@st.cache_data
def load_all_roads():
    import re as _re
    from collections import Counter as _Counter

    _ABBREV_NORMS = [
        (r"\bBOULEVARD\b", "BL"),
        (r"\bBLVD\b",      "BL"),
        (r"\bPARKWAY\b",   "PW"),
        (r"\bPKWY\b",      "PW"),
        (r"\bAVENUE\b",    "AV"),
        (r"\bAVE\b",       "AV"),
        (r"\bDRIVE\b",     "DR"),
        (r"\bROAD\b",      "RD"),
        (r"\bSTREET\b",    "ST"),
        (r"\bLANE\b",      "LN"),
        (r"\bCOURT\b",     "CT"),
        (r"\bPLACE\b",     "PL"),
        (r"\bCIRCLE\b",    "CIR"),
    ]

    def _normalize(raw: str) -> str:
        name = _re.sub(r"^\d+\s+", "", str(raw).strip().upper())
        for pat, repl in _ABBREV_NORMS:
            name = _re.sub(pat, repl, name)
        name = _re.sub(r"[^\w\s]", " ", name)
        name = _re.sub(r"\s+", " ", name).strip()
        return name

    # ── Step 1: frequency-validated names from crash data ──────────────────
    # Correct spellings appear hundreds of times; typos appear 1-2 times.
    # Most-frequent form wins; length is tiebreaker for equal frequency.
    df = pd.read_csv(PROC / "merged_2023.csv", usecols=["road_key"])
    normalized_all = df["road_key"].dropna().map(_normalize)
    freq = _Counter(normalized_all)

    roads = normalized_all.drop_duplicates().sort_values().tolist()

    _JUNK = _re.compile(
        r"^\d+[A-Z]{3,}|^\d+[NSEW]$|\bBLK\b|\bBLOCK\b"
        r"|\bON RAMP\b|\bOFF RAMP\b|\bAT \b|\bEXIT\b"
    )
    roads = [r for r in roads if not _JUNK.search(r) and len(r) >= 4]

    def _is_dir_variant(a: str, b: str) -> bool:
        """True when a and b are the same road but different N/S/E/W segments."""
        _dirs = (" N", " S", " E", " W")
        for d in _dirs:
            if a + d == b or b + d == a:   # one is the other + direction suffix
                return True
        def _strip(name):
            for d in _dirs:
                if name.endswith(d):
                    return name[:-len(d)], True
            return name, False
        a_base, a_has = _strip(a)
        b_base, b_has = _strip(b)
        return a_has and b_has and a_base == b_base  # both directional, same base

    try:
        from rapidfuzz import fuzz as _fuzz
        by_freq = sorted(roads, key=lambda r: (freq.get(r, 0), len(r)), reverse=True)
        to_remove: set[str] = set()
        for i, a in enumerate(by_freq):
            if a in to_remove:
                continue
            for b in by_freq[i + 1:]:
                if b in to_remove:
                    continue
                if _fuzz.ratio(a, b) >= 85 and not _is_dir_variant(a, b):
                    to_remove.add(b)
        clean_roads: set[str] = {r for r in roads if r not in to_remove}
    except ImportError:
        clean_roads = set(roads)

    print(f"[load_all_roads] {len(clean_roads)} frequency-validated road names")

    # ── Step 2: coord filter — keep only roads that have geocoordinates ────
    # Normalise every key in road_coordinates.json; the resulting set is used
    # as a filter so only mappable roads appear in the dropdown.
    coords_raw = json.loads((PROC / "road_coordinates.json").read_text())
    coord_names: set[str] = {_normalize(k) for k in coords_raw.keys()}

    roads = sorted(clean_roads & coord_names)
    print(f"Dropdown updated to geocoded roads only: {len(roads)} roads")
    return roads


@st.cache_data
def load_aerial_b64(mtime: float = 0.0) -> str | None:
    """Read assets/vb_aerial.jpg as base64. mtime param busts cache when file changes."""
    p = BASE / "assets" / "vb_aerial.jpg"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


(
    acc_model, cong_model,
    acc_scaler, cong_scaler,
    acc_feat, cong_feat,
    hotspots, road_coords,
) = load_resources()

all_roads   = load_all_roads()
_aerial_p   = BASE / "assets" / "vb_aerial.jpg"
_aerial_b64 = load_aerial_b64(_aerial_p.stat().st_mtime if _aerial_p.exists() else 0.0)


# ── Utility functions ──────────────────────────────────────────────────────────
def run_model(model, scaler, feat_meta, feat_dict):
    feature_list = feat_meta["features"]
    scaled_cols  = feat_meta["scaled_cols"]
    vec = np.array([[feat_dict[c] for c in feature_list]], dtype=float)
    scaled_idx = [feature_list.index(c) for c in scaled_cols]
    vec[:, scaled_idx] = scaler.transform(vec[:, scaled_idx])
    proba      = model.predict_proba(vec)[0]
    risk_class = ["Low", "Medium", "High"][int(np.argmax(proba))]
    confidence = round(float(np.max(proba)) * 100, 1)
    return risk_class, confidence, proba


_HOTSPOT_KEYS  = None   # populated lazily on first call
_COORD_KEYS    = None

def _resolve_road(selected_road: str, keys: list[str], label: str) -> tuple[str | None, bool]:
    """Return (matched_key, was_fuzzy).  Returns (None, False) if no match >= 80."""
    if selected_road in keys:
        return selected_road, False
    try:
        from rapidfuzz import process as _proc, fuzz as _fuzz
        result = _proc.extractOne(
            selected_road, keys,
            scorer=_fuzz.ratio,
            score_cutoff=80,
        )
        if result:
            matched, score, _ = result
            print(f"[fuzzy-road-lookup] {label}: '{selected_road}' -> '{matched}' (score={score:.0f})")
            return matched, True
    except ImportError:
        pass
    return None, False


def lookup_hotspot(selected_road: str):
    """Return (historical_crash_rate, avg_adt) with fuzzy fallback."""
    global _HOTSPOT_KEYS
    if _HOTSPOT_KEYS is None:
        _HOTSPOT_KEYS = hotspots["road_key"].tolist()

    matched, was_fuzzy = _resolve_road(selected_road, _HOTSPOT_KEYS, "hotspot")
    if matched is not None:
        row = hotspots[hotspots["road_key"] == matched]
        return float(row["total_crashes"].iloc[0]), float(row["avg_ADT"].iloc[0])
    # No match at all — Low-risk defaults
    print(f"[fuzzy-road-lookup] hotspot: '{selected_road}' -> no match, using defaults")
    return 1.0, 8000.0


def lookup_coords(selected_road: str) -> dict | None:
    """Return road_coords entry with fuzzy fallback, or None if no match."""
    global _COORD_KEYS
    if _COORD_KEYS is None:
        _COORD_KEYS = list(road_coords.keys())

    matched, was_fuzzy = _resolve_road(selected_road, _COORD_KEYS, "coords")
    if matched is not None:
        return road_coords[matched]
    print(f"[fuzzy-road-lookup] coords: '{selected_road}' -> no match, skipping star marker")
    return None



# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: LANDING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "landing":

    # ── Load image fresh and inject background on outer Streamlit containers ──
    import base64 as _b64mod, pathlib as _plmod
    _img_path_fresh = _plmod.Path(__file__).parent / "assets" / "vb_aerial.jpg"
    _img_bytes_fresh = _img_path_fresh.read_bytes()
    _img_b64_fresh = _b64mod.b64encode(_img_bytes_fresh).decode()
    print(f"[bg-image] file size: {len(_img_bytes_fresh)/1024:.1f} KB | b64 prefix: {_img_b64_fresh[:30]}")

    st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"],
section[data-testid="stMain"],
.stMainBlockContainer {{
    background-image: url("data:image/jpeg;base64,{_img_b64_fresh}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    min-height: 100vh !important;
}}
</style>
""", unsafe_allow_html=True)
    st.write("")

    # Build background CSS — base64 image embedded directly, 0.45 dark overlay
    if _aerial_b64:
        _bg_layers = (
            f'linear-gradient(rgba(0,0,0,0.45),rgba(0,0,0,0.45)),'
            f'url("data:image/jpeg;base64,{_aerial_b64}")'
        )
        _bg_extra = "background-size:cover;background-position:center;"
    else:
        _bg_layers = "linear-gradient(160deg,#071e3d 0%,#0d2b4e 60%,#0a3060 100%)"
        _bg_extra  = ""

    st.markdown(f"""
<style>
/* Strip all Streamlit chrome on Screen 1 */
[data-testid="stHeader"]  {{ display:none !important; }}
[data-testid="stToolbar"] {{ display:none !important; }}
[data-testid="stSidebar"] {{ display:none !important; }}

/* Make every container transparent and zero-padded */
.stApp,
[data-testid="stAppViewContainer"],
section.main,
[data-testid="stMainBlockContainer"] {{
    background-color: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}}
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
    background-color: transparent !important;
    min-height: 0 !important;
}}

/* Get Started button — fixed at bottom center, above the overlay div */
[data-testid="stButton"] {{
    position: fixed !important;
    bottom: 56px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    z-index: 9999 !important;
    width: 200px !important;
}}
[data-testid="stButton"] > button {{
    background-color: #00A8CC !important;
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    padding: 16px 0 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 22px rgba(0,168,204,0.52) !important;
    width: 200px !important;
    letter-spacing: 0.3px !important;
    white-space: nowrap !important;
}}
[data-testid="stButton"] > button:hover {{
    background-color: #008fad !important;
    border-color: #008fad !important;
    box-shadow: 0 6px 28px rgba(0,168,204,0.70) !important;
}}
</style>

<!-- ── Full-viewport scattered layout ──────────────────────────────────── -->
<div id="lnd-overlay" style="
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    background-image: {_bg_layers};
    {_bg_extra}
    z-index: 10;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    overflow: hidden;
">

  <!-- TOP LEFT ① ocean-blue eyebrow label ──────────────────────────────── -->
  <div style="position:absolute;top:52px;left:64px;">
    <span style="
      color: #00A8CC;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 2.8px;
      text-transform: uppercase;
    ">AI-Powered Risk Intelligence</span>
  </div>

  <!-- TOP LEFT ② main title ────────────────────────────────────────────── -->
  <div style="position:absolute;top:80px;left:64px;max-width:55%;">
    <div style="
      color: #ffffff;
      font-size: 48px;
      font-weight: 800;
      line-height: 1.12;
      text-shadow: 0 2px 18px rgba(0,0,0,0.70), 0 1px 4px rgba(0,0,0,0.55);
    ">Virginia Beach Traffic and Accident Risk Intelligence</div>
  </div>

  <!-- TOP RIGHT accuracy badges ────────────────────────────────────────── -->
  <div style="position:absolute;top:52px;right:64px;display:flex;flex-direction:column;gap:12px;">
    <div style="
      background: rgba(0,0,0,0.58);
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 10px;
      padding: 14px 22px;
      min-width: 235px;
    ">
      <div style="color:#ffffff;font-size:13px;font-weight:600;letter-spacing:0.3px;">
        Accident Prediction
      </div>
      <div style="color:#00A8CC;font-size:16px;font-weight:700;margin-top:5px;">
        Moderate accuracy
      </div>
    </div>
    <div style="
      background: rgba(0,0,0,0.58);
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 10px;
      padding: 14px 22px;
      min-width: 235px;
    ">
      <div style="color:#ffffff;font-size:13px;font-weight:600;letter-spacing:0.3px;">
        Congestion Prediction
      </div>
      <div style="color:#00A8CC;font-size:16px;font-weight:700;margin-top:5px;">
        Moderate accuracy
      </div>
    </div>
  </div>

  <!-- MIDDLE CENTER warning pill ───────────────────────────────────────── -->
  <div style="
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    white-space: nowrap;
  ">
    <span style="
      background: #FFD700;
      color: #1a1a1a;
      font-size: 13px;
      font-weight: 700;
      padding: 5px 14px;
      border-radius: 20px;
      display: inline-block;
      margin-right: 9px;
      vertical-align: middle;
    ">Please note:</span>
    <span style="
      color: #ffffff;
      font-size: 15px;
      font-weight: 600;
      vertical-align: middle;
      text-shadow: 0 1px 6px rgba(0,0,0,0.55);
    ">Predictions are only available for trips within the next 5 days</span>
  </div>

  <!-- BOTTOM LEFT description ──────────────────────────────────────────── -->
  <div style="position:absolute;bottom:130px;left:64px;max-width:45%;">
    <p style="
      color: #ffffff;
      font-size: 16px;
      font-weight: 500;
      line-height: 1.68;
      margin: 0;
      text-shadow: 0 1px 8px rgba(0,0,0,0.60);
    ">Real-time accident and congestion risk predictions powered by
    machine learning and live weather data</p>
  </div>


</div>
""", unsafe_allow_html=True)

    if st.button("Get Started →", key="get_started", type="primary"):
        st.session_state.screen = "prediction"
        st.rerun()

    print("Button centered")


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
else:

    # ── Screen 2 styles ────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        /* ── Page & layout ───────────────────────────────────── */
        .stApp,
        [data-testid="stAppViewContainer"],
        section.main,
        [data-testid="stMainBlockContainer"]
                                { background-color: #0B4F6C !important; }
        .block-container        { background-color: transparent !important;
                                  padding-top: 0.5rem !important;
                                  padding-bottom: 0.5rem !important; }

        /* ── Hide sidebar, toolbar, and header bar ──────────── */
        [data-testid="stSidebar"] { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stHeader"]  { display: none !important; }

        /* ── Headings ─────────────────────────────────────────── */
        h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }

        /* ── Paragraph / written text ────────────────────────── */
        [data-testid="stText"],
        [data-testid="stText"] *,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span  { color: #ffffff !important; }

        /* ── Widget labels (form field names) ────────────────── */
        label                                     { color: #ffffff !important;
                                                    font-weight: 700 !important; }
        div[data-testid="stWidgetLabel"] p        { color: #ffffff !important;
                                                    font-weight: 700 !important; }

        /* ── st.write() text ─────────────────────────────────── */
        [data-testid="stVerticalBlock"] > div > p { color: #ffffff !important; }

        /* ── Caption text ────────────────────────────────────── */
        [data-testid="stCaptionContainer"] p      { color: rgba(255,255,255,0.70) !important; }

        /* ── Form card ───────────────────────────────────────── */
        [data-testid="stForm"] {
            background    : rgba(0,0,0,0.22) !important;
            border        : 1px solid rgba(255,255,255,0.22) !important;
            border-radius : 10px !important;
            padding       : 14px !important;
        }

        /* ── Selectbox control — white bg, dark text ─────────── */
        .stSelectbox [data-baseweb="select"] > div,
        .stSelectbox [data-baseweb="select"] > div > div {
            background-color : #ffffff !important;
            border-color     : #bbbbbb !important;
        }
        .stSelectbox [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] div[class*="singleValue"],
        .stSelectbox [data-baseweb="select"] div[class*="placeholder"],
        .stSelectbox [data-baseweb="select"] input  { color: #1a1a1a !important; }
        .stSelectbox [data-baseweb="select"] svg    { fill: #444444 !important; }

        /* ── Dropdown popup list — white bg, dark text ─────────── */
        [data-baseweb="popover"] > div,
        [data-baseweb="popover"] [data-baseweb="menu"],
        ul[data-baseweb="menu"]                 { background-color: #ffffff !important;
                                                  border-color: #cccccc !important; }
        [data-baseweb="option"]                 { background-color: #ffffff !important;
                                                  color: #1a1a1a !important; }
        [data-baseweb="option"]:hover           { background-color: #e0f2f7 !important;
                                                  color: #0B4F6C !important; }
        [data-baseweb="option"][aria-selected="true"] {
                                                  background-color: #cceaf4 !important;
                                                  color: #0B4F6C !important; }

        /* ── Date input — white bg, dark text ───────────────── */
        .stDateInput input {
            background-color : #ffffff !important;
            border-color     : #bbbbbb !important;
            color            : #1a1a1a !important;
        }
        .stDateInput button,
        .stDateInput svg                        { color: #333333 !important;
                                                  fill:  #333333 !important; }

        /* ── Back button — white text, visible border ────────── */
        div[data-testid="stButton"] > button:not([kind="primary"]) {
            color            : #ffffff !important;
            border           : 1.5px solid rgba(255,255,255,0.65) !important;
            background-color : rgba(255,255,255,0.10) !important;
        }
        div[data-testid="stButton"] > button:not([kind="primary"]):hover {
            background-color : rgba(255,255,255,0.20) !important;
            border-color     : #ffffff !important;
        }

        /* ── Analyze Risk button — ocean blue, white text ───── */
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stButton"] > button[kind="primary"] {
            background-color : #00A8CC !important;
            color            : #ffffff !important;
            border-color     : #00A8CC !important;
        }

        /* ── Weather info banner ─────────────────────────────── */
        [data-testid="stAlert"],
        [data-testid="stInfo"]                  { background-color: rgba(0,25,45,0.85) !important;
                                                  border-color: #00A8CC !important; }
        [data-testid="stAlert"] *,
        [data-testid="stInfo"] *                { color: #ffffff !important; }
        [data-testid="stWarning"]               { background-color: rgba(244,162,97,0.20) !important; }
        [data-testid="stWarning"] *             { color: #ffffff !important; }

        /* ── HR ─────────────────────────────────────────────── */
        hr { border-color: rgba(255,255,255,0.20) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Back button ────────────────────────────────────────────────────────────
    if st.button("← Back", key="back_btn"):
        st.session_state.screen = "landing"
        st.session_state.submitted = False
        st.session_state.results = None
        st.rerun()

    # ── Weather banner — full width, above columns ─────────────────────────────
    if st.session_state.submitted and st.session_state.results is not None:
        _r = st.session_state.results
        if _r["weather_ok"] and _r["weather_temp_f"] is not None:
            st.info(
                f"**Weather for {_r['date_str']} at {_r['display_time']}:** "
                f"{_r['weather_desc']} · {_r['weather_temp_f']:.0f}°F · "
                f"Wind {_r['weather_wind']:.1f} mph"
            )
        else:
            st.info(
                f"**Conditions for {_r['date_str']} at {_r['display_time']}:** "
                f"Clear · No weather data available"
            )

    col_left, col_right = st.columns([4, 6])

    # ── Left column — input form ───────────────────────────────────────────────
    with col_left:
        st.header("Enter Your Trip Details")

        today    = date.today()
        tomorrow = today + timedelta(days=1)
        max_date = today + timedelta(days=5)

        with st.form("trip_form"):
            selected_road = st.selectbox("Select Road", options=all_roads)

            selected_date = st.date_input(
                "Travel Date",
                value=tomorrow,
                min_value=today,
                max_value=max_date,
            )

            st.write("Departure Time")
            tc = st.columns(3)
            with tc[0]:
                sel_hour_12 = st.selectbox(
                    "Hour",
                    options=list(range(1, 13)),
                    format_func=str,
                    index=4,          # default: 5
                )
            with tc[1]:
                sel_minute = st.selectbox(
                    "Min",
                    options=list(range(60)),
                    format_func=lambda m: f"{m:02d}",
                    index=0,
                )
            with tc[2]:
                sel_ampm = st.selectbox(
                    "AM / PM",
                    options=["AM", "PM"],
                    index=1,          # default: PM
                )

            submitted = st.form_submit_button(
                "Analyze Risk",
                type="primary",
                use_container_width=True,
            )

    # ── Convert AM/PM → 24-hour; round to nearest hour ────────────────────────
    _min = int(sel_minute)
    if sel_ampm == "PM" and sel_hour_12 < 12:
        hour_24 = sel_hour_12 + 12
    elif sel_ampm == "AM" and sel_hour_12 == 12:
        hour_24 = 0
    else:
        hour_24 = sel_hour_12

    rounded_hour = (hour_24 + 1) % 24 if _min >= 30 else hour_24
    display_time = f"{sel_hour_12}:{_min:02d} {sel_ampm}"

    # ── Post-submit compute pipeline ───────────────────────────────────────────
    if submitted:

        if selected_date < today or selected_date > max_date:
            st.error(
                f"Invalid travel date. Please choose a date between "
                f"**{today.strftime('%B %d, %Y')}** and "
                f"**{max_date.strftime('%B %d, %Y')}**."
            )
            st.stop()

        # Weather API
        WEATHER_ENCODE = {
            "Clear": 0, "Clouds": 0,
            "Mist": 1, "Drizzle": 1,
            "Rain": 2, "Thunderstorm": 2,
            "Fog": 3,
            "Snow": 4,
        }

        api_key         = OPENWEATHER_API_KEY
        weather_encoded = 0
        weather_desc    = "Clear"
        weather_temp_f  = None
        weather_wind    = None
        weather_ok      = False

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
            forecast_list = resp.json()["list"]

            target_ts = datetime(
                selected_date.year, selected_date.month, selected_date.day,
                rounded_hour, 0, 0,
            ).timestamp()

            best            = min(forecast_list, key=lambda e: abs(e["dt"] - target_ts))
            weather_main    = best["weather"][0]["main"]
            weather_desc    = best["weather"][0]["description"].capitalize()
            weather_temp_f  = best["main"]["temp"]
            weather_wind    = best["wind"]["speed"]
            weather_encoded = WEATHER_ENCODE.get(weather_main, 0)
            weather_ok      = True

        except Exception:
            st.warning(
                "Weather data is temporarily unavailable. "
                "Predictions assume clear, dry conditions."
            )

        # Feature vector — exact match then fuzzy fallback
        historical_crash_rate, avg_adt = lookup_hotspot(selected_road)

        is_night    = 1 if (rounded_hour < 6 or rounded_hour >= 20) else 0
        is_rush     = 1 if rounded_hour in [7, 8, 9, 16, 17, 18]   else 0
        is_weekend  = 1 if selected_date.weekday() in [5, 6]        else 0
        school_zone = 1 if (7 <= rounded_hour <= 16 and not is_weekend) else 0
        light_enc   = 0 if (6 <= rounded_hour <= 19) else 2
        surface_enc = 1 if weather_encoded >= 2 else 0

        full_feat = {
            "hour":                   float(rounded_hour),
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

        acc_class,  acc_conf,  _ = run_model(acc_model,  acc_scaler,  acc_feat,  full_feat)
        cong_class, cong_conf, _ = run_model(cong_model, cong_scaler, cong_feat, full_feat)

        # Contributing factors — kept for explanation, not displayed
        factors: list[str] = []
        if is_night:
            factors.append("poor lighting conditions (nighttime)")
        if weather_encoded >= 2:
            factors.append("adverse weather forecast")
        if historical_crash_rate >= 5:
            factors.append("historically high crash rate on this corridor")
        if is_rush:
            factors.append("rush hour traffic window")
        if avg_adt > 36_000:
            factors.append("high daily traffic volume corridor")
        if full_feat["at_intersection_binary"] == 1:
            factors.append("intersection complexity")
        if surface_enc == 1:
            factors.append("wet road surface expected")

        # ── Build structured explanation ───────────────────────────────────────
        day_name   = selected_date.strftime("%A")
        month_name = selected_date.strftime("%B")
        day_num    = selected_date.day
        date_str   = selected_date.strftime("%A, %B %d")

        # Sentence 1
        sent1 = (
            f"Your trip on {selected_road} is forecast to carry "
            f"{acc_class.lower()} accident risk and {cong_class.lower()} congestion risk "
            f"for {day_name}, {month_name} {day_num} at {display_time}."
        )

        # Sentence 2 — factors
        if not factors:
            sent2 = "No significant risk factors were identified for these conditions."
        elif len(factors) == 1:
            sent2 = f"The primary concern is {factors[0]}."
        elif len(factors) == 2:
            sent2 = f"The primary concerns are {factors[0]} and {factors[1]}."
        else:
            factor_list = ", ".join(factors[:-1]) + f", and {factors[-1]}"
            sent2 = f"The primary concerns are {factor_list}."

        # Sentence 3 — safety tip by risk level
        if acc_class == "Low" and cong_class == "Low":
            sent3 = "Conditions look good for your trip. Drive safely and enjoy the road."
        elif acc_class == "Low" and cong_class == "Medium":
            sent3 = (
                "Accident risk is low but expect some traffic on this corridor. "
                "Allow a few extra minutes for your trip."
            )
        elif acc_class == "Low" and cong_class == "High":
            sent3 = (
                "Accident risk is low but heavy congestion is expected. "
                "Consider traveling at a different time to avoid delays."
            )
        elif acc_class == "Medium":
            sent3 = (
                "We recommend reducing speed, staying alert, and allowing extra "
                "travel time for this trip."
            )
        elif acc_class == "High" and cong_class == "Low":
            sent3 = (
                "Driving conditions are hazardous on this corridor. Strongly consider "
                "adjusting your departure time or taking an alternate route."
            )
        else:
            sent3 = (
                "Driving conditions are hazardous and traffic will be heavy. "
                "We strongly recommend postponing this trip or choosing a safer "
                "alternate route."
            )

        fallback_text = f"{sent1} {sent2} {sent3}"

        # Groq prompt — ask it to follow the exact 3-sentence pattern
        groq_prompt = (
            f"Write exactly 3 sentences following this structure:\n\n"
            f"Sentence 1 (copy verbatim): \"{sent1}\"\n\n"
            f"Sentence 2 (factors): Risk factors present: "
            f"{', '.join(factors) if factors else 'none'}. "
            f"If none, write: 'No significant risk factors were identified for these conditions.' "
            f"If one factor, write: 'The primary concern is [factor].' "
            f"If multiple, write: 'The primary concerns are [factor1], [factor2], and [factorN].'\n\n"
            f"Sentence 3 (copy verbatim): \"{sent3}\"\n\n"
            f"Rules: plain English, no bullet points, no traffic signals, output 3 sentences only."
        )

        ai_text: str | None = None
        groq_key = GROQ_API_KEY
        if _GROQ_AVAILABLE and groq_key and groq_key != "your_groq_key_here":
            try:
                _gclient    = _Groq(api_key=groq_key)
                _completion = _gclient.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": groq_prompt}],
                    max_tokens=220,
                    temperature=0.3,
                )
                ai_text = _completion.choices[0].message.content.strip()
            except Exception:
                pass

        if ai_text is None:
            ai_text = fallback_text

        # Map sort column
        if is_night:
            sort_col   = "night_crashes"
            time_label = "night (8 pm - 6 am)"
        elif is_rush:
            sort_col   = "rush_crashes"
            time_label = "rush hour (7-9 am, 4-6 pm)"
        else:
            sort_col   = "total_crashes"
            time_label = "daytime"

        # Store results
        st.session_state.results = {
            "selected_road":  selected_road,
            "date_str":       date_str,
            "display_time":   display_time,
            "weather_ok":     weather_ok,
            "weather_desc":   weather_desc,
            "weather_temp_f": weather_temp_f,
            "weather_wind":   weather_wind,
            "acc_class":      acc_class,
            "acc_conf":       acc_conf,
            "cong_class":     cong_class,
            "cong_conf":      cong_conf,
            "ai_text":        ai_text,
            "sort_col":       sort_col,
            "time_label":     time_label,
        }
        st.session_state.submitted = True

    # ── Results: risk cards + explanation (right col) then full-width map ────────
    if st.session_state.submitted and st.session_state.results is not None:
        r = st.session_state.results

        with col_right:
            _COLORS = {"High": "#E63946", "Medium": "#F4A261", "Low": "#0ACF83"}
            card_a, card_b = st.columns(2)

            def _risk_card(col, label, risk_class, confidence):
                bg = _COLORS.get(risk_class, "#888")
                col.markdown(
                    f"""
                    <div style="background:{bg};padding:12px 10px;border-radius:10px;
                                text-align:center;color:white;margin-bottom:6px;">
                        <div style="font-size:12px;font-weight:700;letter-spacing:1px;
                                    margin-bottom:5px;">{label}</div>
                        <div style="font-size:28px;font-weight:800;line-height:1.1;">
                            {risk_class}
                        </div>
                        <div style="font-size:13px;margin-top:5px;opacity:0.92;">
                            {confidence:.0f}% confident
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            _risk_card(card_a, "ACCIDENT RISK",   r["acc_class"], r["acc_conf"])
            _risk_card(card_b, "CONGESTION RISK", r["cong_class"], r["cong_conf"])

            # Explanation
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.10);border-left:4px solid #0099CC;
                            padding:10px 14px;border-radius:8px;margin-bottom:10px;">
                    <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:#ffffff;">
                        Explanation
                    </div>
                    <div style="font-size:13px;line-height:1.60;color:rgba(255,255,255,0.92);">
                        {r['ai_text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Map — all roads with coordinates, colored by risk ─────────────
            _CIRCLE_COLORS = {
                "High":   "#E63946",
                "Medium": "#F4A261",
                "Low":    "#0ACF83",
            }

            # All roads that have coordinates
            mapped_roads = hotspots[hotspots["lat"].notna()].copy()

            # Center on selected road if coords available, else VB center
            rc = lookup_coords(r["selected_road"])
            map_center = [rc["lat"], rc["lon"]] if rc else [36.8529, -75.9780]

            fmap = folium.Map(
                location=map_center,
                zoom_start=12,
                tiles="CartoDB positron",
            )

            for _, hrow in mapped_roads.iterrows():
                level = hrow["accident_risk_level"]
                color = _CIRCLE_COLORS.get(level, "#888888")
                popup_html = (
                    f"<b>{hrow['road_key']}</b><br>"
                    f"Crashes (2023): {int(hrow['total_crashes'])}<br>"
                    f"Accident Risk: {level}<br>"
                    f"Congestion Risk: {hrow['congestion_risk_level']}<br>"
                    f"Avg ADT: {int(hrow['avg_ADT']):,}"
                )
                folium.CircleMarker(
                    location=[hrow["lat"], hrow["lon"]],
                    radius=7,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.72,
                    weight=1.5,
                    popup=folium.Popup(popup_html, max_width=230),
                    tooltip=f"{hrow['road_key']} — {level} Risk",
                ).add_to(fmap)

            # Star marker for selected road
            if rc is not None:
                folium.Marker(
                    location=[rc["lat"], rc["lon"]],
                    icon=folium.DivIcon(
                        html=(
                            '<div style="font-size:28px;line-height:1;'
                            'text-align:center;color:#1565C0;">&#11088;</div>'
                        ),
                        icon_size=(32, 32),
                        icon_anchor=(16, 16),
                    ),
                    popup=folium.Popup("Your selected road", max_width=160),
                    tooltip="Your selected road",
                ).add_to(fmap)

            legend_html = """
            <div style="position:fixed;bottom:30px;right:30px;z-index:9999;
                        background:white;padding:10px 14px;
                        border:2px solid #bbb;border-radius:8px;
                        font-size:12px;line-height:1.9;
                        box-shadow:2px 2px 8px rgba(0,0,0,0.18);">
                &#x1F7E2; Low risk<br>
                &#x1F7E0; Medium risk<br>
                &#x1F534; High risk<br>
                &#11088; Your selected road
            </div>
            """
            fmap.get_root().html.add_child(folium.Element(legend_html))

            st_folium(fmap, use_container_width=True, height=250)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:rgba(255,255,255,0.18);margin-top:24px;'>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Data: Virginia Beach Police Department 2023 · "
        "Weather: OpenWeatherMap · "
        "AI: Groq / Llama 3 · "
        "Model: XGBoost · "
        "Predictions valid within 5-day forecast window only"
    )
