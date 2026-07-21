import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
PROC    = BASE / "data" / "processed"
FIGURES = BASE / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── load (prefer locked original; fall back to _new) ──────────────────────
for fname in ["merged_2023.csv", "merged_2023_new.csv"]:
    fpath = PROC / fname
    if fpath.exists():
        try:
            df_raw = pd.read_csv(fpath)
            print(f"Loaded {fname}  —  {len(df_raw):,} rows")
            break
        except PermissionError:
            continue

df = df_raw[df_raw["match_type"] != "median_fill"].copy()
print(f"After dropping median_fill rows: {len(df):,} rows\n")

# ── style ─────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

C_NIGHT = "#4e79a7"   # blue  – night shading & night line
C_RUSH  = "#f28e2b"   # amber – rush-hour shading & day line
C_BAR   = "#59a14f"   # green – generic bars
C_BAR2  = "#76b7b2"   # teal  – secondary bars

NIGHT_HOURS = list(range(0, 6))  + list(range(20, 24))
RUSH_HOURS  = list(range(7, 10)) + list(range(16, 19))
DAY_HOURS   = list(range(6, 20))

SEP = "=" * 65

def save(fig: plt.Figure, name: str) -> None:
    out = FIGURES / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ══════════════════════════════════════════════════════════════════════════
# 1. Crashes by hour of day
# ══════════════════════════════════════════════════════════════════════════
hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(hourly.index, hourly.values, color=C_BAR, zorder=3)

# shaded regions
for start, end in [(0, 5.5), (19.5, 23.5)]:
    ax.axvspan(start - 0.5, end, alpha=0.12, color=C_NIGHT, label="_nolegend_", zorder=0)
for start, end in [(6.5, 9.5), (15.5, 18.5)]:
    ax.axvspan(start, end, alpha=0.12, color=C_RUSH, label="_nolegend_", zorder=0)

night_patch = mpatches.Patch(color=C_NIGHT, alpha=0.35, label="Night (0–5, 20–23)")
rush_patch  = mpatches.Patch(color=C_RUSH,  alpha=0.35, label="Rush hours (7–9, 16–18)")
ax.legend(handles=[night_patch, rush_patch], loc="upper left")

ax.set_xticks(range(24))
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Crash Count")
ax.set_title("Crash Frequency by Hour of Day", fontsize=14, fontweight="bold")
save(fig, "01_crashes_by_hour.png")

peak_hour   = int(hourly.idxmax())
peak_count  = int(hourly.max())
night_total = int(hourly[NIGHT_HOURS].sum())
night_pct   = night_total / len(df) * 100


# ══════════════════════════════════════════════════════════════════════════
# 2. Crashes by day of week
# ══════════════════════════════════════════════════════════════════════════
DOW_ORDER = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
dow_counts = (
    df["Day_of_Week"]
    .str.strip().str.upper()
    .value_counts()
    .reindex(DOW_ORDER, fill_value=0)
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(dow_counts.index, dow_counts.values, color=C_BAR2, zorder=3)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Crash Count")
ax.set_title("Crash Frequency by Day of Week", fontsize=14, fontweight="bold")
save(fig, "02_crashes_by_dow.png")

busiest_day = dow_counts.idxmax()
quietest_day = dow_counts.idxmin()


# ══════════════════════════════════════════════════════════════════════════
# 3. Crashes by weather condition
# ══════════════════════════════════════════════════════════════════════════
wx = df["Weather_Condition"].dropna().str.strip().value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(wx.index, wx.values, color="#e15759", zorder=3)
ax.set_xlabel("Weather Condition")
ax.set_ylabel("Crash Count")
ax.set_title("Crash Frequency by Weather Condition", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
save(fig, "03_crashes_by_weather.png")

clear_pct = wx.iloc[0] / wx.sum() * 100
top_wx    = wx.index[0]


# ══════════════════════════════════════════════════════════════════════════
# 4. Crashes by light condition
# ══════════════════════════════════════════════════════════════════════════
light = df["Light_Conditions"].dropna().str.strip().value_counts()

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(light.index, light.values, color="#edc948", zorder=3)
ax.set_xlabel("Light Condition")
ax.set_ylabel("Crash Count")
ax.set_title("Crash Frequency by Light Condition", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
save(fig, "04_crashes_by_light.png")

top_light     = light.index[0]
top_light_pct = light.iloc[0] / light.sum() * 100
dark_cols = [c for c in light.index if "DARK" in c.upper()]
dark_total = light[dark_cols].sum() if dark_cols else 0
dark_pct   = dark_total / light.sum() * 100


# ══════════════════════════════════════════════════════════════════════════
# 5. Crashes by road surface condition
# ══════════════════════════════════════════════════════════════════════════
surf = df["Roadway_Surface_Condition"].dropna().str.strip().value_counts()

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(surf.index, surf.values, color="#b07aa1", zorder=3)
ax.set_xlabel("Road Surface Condition")
ax.set_ylabel("Crash Count")
ax.set_title("Crash Frequency by Road Surface Condition", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
save(fig, "05_crashes_by_surface.png")

dry_pct = surf.get("DRY", 0) / surf.sum() * 100
wet_pct = surf.get("WET", 0) / surf.sum() * 100


# ══════════════════════════════════════════════════════════════════════════
# 6. Top 15 roads by crash count
# ══════════════════════════════════════════════════════════════════════════
top15 = df["road_key"].value_counts().head(15).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top15.index, top15.values, color=C_BAR, zorder=3)
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_xlabel("Crash Count")
ax.set_title("Top 15 Roads by Crash Count (2023)", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "06_top15_roads.png")

top_road       = top15.index[-1]
top_road_count = int(top15.values[-1])


# ══════════════════════════════════════════════════════════════════════════
# 7. ADT distribution  (y = number of crashes)
# ══════════════════════════════════════════════════════════════════════════
adt = df["ADT"].dropna()
p33 = np.percentile(adt, 33)
p66 = np.percentile(adt, 66)

low_pct = (adt <  p33).sum() / len(adt) * 100
med_pct = ((adt >= p33) & (adt < p66)).sum() / len(adt) * 100
hi_pct  = (adt >= p66).sum() / len(adt) * 100

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(adt, bins=20, color=C_BAR, edgecolor="white", zorder=3)
ax.axvline(p33, color="navy",   linestyle="--", linewidth=1.8,
           label=f"33rd pct  ADT={p33:,.0f}")
ax.axvline(p66, color="crimson", linestyle="--", linewidth=1.8,
           label=f"66th pct  ADT={p66:,.0f}")

ymax = ax.get_ylim()[1]
ax.text(p33 / 2,            ymax * 0.88, f"Low\n{low_pct:.1f}%",
        ha="center", fontsize=10, color="navy",   fontweight="bold")
ax.text((p33 + p66) / 2,   ymax * 0.88, f"Medium\n{med_pct:.1f}%",
        ha="center", fontsize=10, color="darkorange", fontweight="bold")
ax.text(p66 + (adt.max() - p66) / 2, ymax * 0.88, f"High\n{hi_pct:.1f}%",
        ha="center", fontsize=10, color="crimson", fontweight="bold")

ax.legend()
ax.set_xlabel("Average Daily Traffic (ADT)")
ax.set_ylabel("Number of Crashes")
ax.set_title("Distribution of Average Daily Traffic (ADT)", fontsize=14, fontweight="bold")
save(fig, "07_adt_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# 8. Night vs day crash patterns by hour
# ══════════════════════════════════════════════════════════════════════════
night_series = hourly.where(hourly.index.isin(NIGHT_HOURS))
day_series   = hourly.where(hourly.index.isin(DAY_HOURS))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(night_series.index, night_series.values, "o-", color=C_NIGHT,
        linewidth=2.2, markersize=6, label="Night (0–5, 20–23)")
ax.plot(day_series.index, day_series.values, "s-", color=C_RUSH,
        linewidth=2.2, markersize=6, label="Day (6–19)")
ax.set_xticks(range(24))
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Crash Count")
ax.set_title("Night vs Day Crash Patterns by Hour", fontsize=14, fontweight="bold")
ax.legend()
save(fig, "08_night_vs_day.png")


# ══════════════════════════════════════════════════════════════════════════
# 9. Correlation heatmap
# ══════════════════════════════════════════════════════════════════════════
NUM_COLS = ["hour", "Number_of_Vehicles_Involved", "ADT"]
if "AAWDT" in df.columns:
    NUM_COLS.append("AAWDT")
if "signal_count" in df.columns:
    NUM_COLS.append("signal_count")

corr = df[NUM_COLS].corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "09_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# 10. Crash count distribution — integer bins, last bar is "15+"
# ══════════════════════════════════════════════════════════════════════════
road_crashes = df.groupby("road_key").size()

cr_min    = int(road_crashes.min())
cr_max    = int(road_crashes.max())
cr_mean   = road_crashes.mean()
cr_median = road_crashes.median()
cr_p33    = np.percentile(road_crashes, 33)
cr_p50    = np.percentile(road_crashes, 50)
cr_p66    = np.percentile(road_crashes, 66)
cr_p90    = np.percentile(road_crashes, 90)

CLIP = 15
bins   = list(range(1, CLIP + 1))          # 1 … 14 exact, 15 = "15+"
counts = [int((road_crashes == v).sum()) for v in range(1, CLIP)]
counts.append(int((road_crashes >= CLIP).sum()))
labels = [str(v) for v in range(1, CLIP)] + ["15+"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(bins)), counts, color="#9c755f", edgecolor="white", zorder=3)
ax.bar_label(bars, labels=[str(c) for c in counts],
             padding=3, fontsize=8, fontweight="bold")

ax.set_xticks(range(len(bins)))
ax.set_xticklabels(labels)
ax.set_xlabel("Number of Crashes per Road in 2023")
ax.set_ylabel("Number of Roads")
ax.set_title("Crash Count Distribution Across Virginia Beach Roads (2023)",
             fontsize=14, fontweight="bold")
ax.set_ylim(0, max(counts) * 1.12)

stats_text = (
    f"Min:    {cr_min}\n"
    f"Max:    {cr_max}\n"
    f"Mean:   {cr_mean:.1f}\n"
    f"Median: {cr_median:.1f}"
)
ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85))
plt.tight_layout()
save(fig, "10_crash_rate_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# 11. Top 20 roads by crash count (with visible counts)
# ══════════════════════════════════════════════════════════════════════════
top20 = df["road_key"].value_counts().head(20).sort_values()

fig, ax = plt.subplots(figsize=(11, 8))
bars = ax.barh(top20.index, top20.values, color=C_BAR, zorder=3)
ax.bar_label(bars, labels=[str(v) for v in top20.values],
             padding=4, fontsize=9, fontweight="bold")
ax.set_xlim(0, top20.max() * 1.14)
ax.set_xlabel("Crash Count")
ax.set_title("Top 20 Roads by Crash Count (2023)", fontsize=14, fontweight="bold")
plt.tight_layout()
save(fig, "11_top20_roads_crash_count.png")


# ══════════════════════════════════════════════════════════════════════════
# Key findings summary
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("KEY FINDINGS SUMMARY")
print(SEP)

print(f"\n[1] Crashes by Hour")
print(f"    Peak crash hour : {peak_hour}:00  ({peak_count} crashes)")
print(f"    Night-hour share: {night_pct:.1f}% of all crashes occur 0–5 or 20–23")

print(f"\n[2] Crashes by Day of Week")
print(f"    Busiest day : {busiest_day}  ({dow_counts[busiest_day]:,} crashes)")
print(f"    Quietest day: {quietest_day}  ({dow_counts[quietest_day]:,} crashes)")
weekday_total  = dow_counts[["MON","TUE","WED","THU","FRI"]].sum()
weekend_total  = dow_counts[["SAT","SUN"]].sum()
print(f"    Weekday crashes: {weekday_total:,} vs Weekend: {weekend_total:,}")

print(f"\n[3] Crashes by Weather")
print(f"    '{top_wx}' is the top condition — {clear_pct:.1f}% of weather-coded crashes")
bad_wx_pct = (1 - wx.iloc[0] / wx.sum()) * 100
print(f"    Adverse weather accounts for {bad_wx_pct:.1f}% of weather-coded crashes")

print(f"\n[4] Crashes by Light Condition")
print(f"    '{top_light}' is the top condition — {top_light_pct:.1f}% of light-coded crashes")
print(f"    All darkness conditions combined: {dark_pct:.1f}%")

print(f"\n[5] Crashes by Road Surface")
print(f"    Dry surface: {dry_pct:.1f}%  |  Wet surface: {wet_pct:.1f}%")

print(f"\n[6] Top 15 Roads")
print(f"    Highest-crash road: '{top_road}'  ({top_road_count} crashes in 2023)")

print(f"\n[7] ADT Distribution")
print(f"    Low  congestion  (ADT < {p33:,.0f})  —  {low_pct:.1f}% of crashes")
print(f"    Med  congestion  (ADT {p33:,.0f} – {p66:,.0f})  —  {med_pct:.1f}% of crashes")
print(f"    High congestion  (ADT > {p66:,.0f})  —  {hi_pct:.1f}% of crashes")

night_sum = int(hourly[NIGHT_HOURS].sum())
day_sum   = int(hourly[DAY_HOURS].sum())
print(f"\n[8] Night vs Day")
print(f"    Night (0–5, 20–23): {night_sum:,} crashes  ({night_sum/(night_sum+day_sum)*100:.1f}%)")
print(f"    Day   (6–19):       {day_sum:,} crashes  ({day_sum/(night_sum+day_sum)*100:.1f}%)")

print(f"\n[9] Correlation Matrix")
adt_hour_corr = corr.loc["ADT", "hour"] if "ADT" in corr.index and "hour" in corr.columns else None
adt_veh_corr  = corr.loc["ADT", "Number_of_Vehicles_Involved"] if "ADT" in corr.index else None
if adt_hour_corr is not None:
    print(f"    ADT vs hour:               {adt_hour_corr:+.2f}")
if adt_veh_corr is not None:
    print(f"    ADT vs Number_of_Vehicles: {adt_veh_corr:+.2f}")
if "AAWDT" in corr.index:
    print(f"    ADT vs AAWDT:              {corr.loc['ADT','AAWDT']:+.2f}  (expected near 1.0)")

print(f"\n[10] Crash Rate Distribution")
print(f"    Roads in dataset: {len(road_crashes):,}")
print(f"    Min crashes/road: {cr_min}    Max: {cr_max}")
print(f"    Mean: {cr_mean:.1f}    Median: {cr_median:.1f}")
print(f"    p33={cr_p33:.1f}  p50={cr_p50:.1f}  p66={cr_p66:.1f}  p90={cr_p90:.1f}")
single_crash_roads = (road_crashes == 1).sum()
print(f"    Roads with only 1 crash: {single_crash_roads} ({single_crash_roads/len(road_crashes)*100:.1f}%)")

print(f"\n[11] Top 20 Roads")
top_road20       = top20.index[-1]
top_road20_count = int(top20.values[-1])
print(f"    #1 road: '{top_road20}'  ({top_road20_count} crashes)")
print(f"    #20 road: '{top20.index[0]}'  ({int(top20.values[0])} crashes)")

print(f"\n{SEP}")
print("CRASH COUNT PER ROAD — FULL STATISTICS")
print(SEP)
print(f"  Min:    {cr_min}")
print(f"  Max:    {cr_max}")
print(f"  Mean:   {cr_mean:.2f}")
print(f"  Median: {cr_median:.1f}")
print(f"  p33:    {cr_p33:.1f}")
print(f"  p50:    {cr_p50:.1f}")
print(f"  p66:    {cr_p66:.1f}")
print(f"  p90:    {cr_p90:.1f}")

print(f"\n{SEP}")
print(f"All 11 figures saved to {FIGURES}")
print(SEP)
