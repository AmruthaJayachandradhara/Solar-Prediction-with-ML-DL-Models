"""
Solar PV Data Cleaning Pipeline
================================
Location: Rajasthan, India  (lat=26.810578, lon=73.768455)
Sources:  PVGIS hourly 2019-2020 | NASA POWER 2019-2020 | PVGIS TMY
Target:   processed_solar_2019_2020.csv  — ready for PV power prediction

Pipeline Stages
---------------
  1. Load & parse timestamps for every source
  2. Compute derived columns (G_i) and unit conversions
  3. Time-align PVGIS (:30) → (:00) so it matches NASA (:00)
  4. Merge NASA onto PVGIS hourly  → adds RH, SP
  5. Merge TMY onto combined data  → adds WD10m (by month-day-hour pattern)
  6. Sentinel / missing-value handling
  7. Validate & export
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE = Path("/Users/amruthaj/Desktop/Desktop/Capstone")
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "datacleaning"

PVGIS_HOURLY   = DATA_DIR / "raw_pvgis_hourly_2019_2020.csv"
NASA_2019      = DATA_DIR / "raw_nasa_2019.csv"
NASA_2020      = DATA_DIR / "raw_nasa_2020.csv"
PVGIS_TMY      = DATA_DIR / "raw_pvgis_tmy.csv"
OUTPUT         = OUT_DIR / "processed_solar_2019_2020.csv"
REPORT         = OUT_DIR / "cleaning_report.txt"

# ──────────────────────────────────────────────────────────────────────────────
# SENTINEL VALUES  (mark as NaN before any calculation)
# ──────────────────────────────────────────────────────────────────────────────
SENTINELS = [-999.0, -9999.0, -99.0]


def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common API sentinel values with NaN."""
    return df.replace(SENTINELS, np.nan)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — LOAD & PARSE  ─ PVGIS HOURLY 2019-2020
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STAGE 1 — Loading PVGIS hourly 2019-2020")
print("=" * 60)

pvgis = pd.read_csv(PVGIS_HOURLY)
pvgis.columns = pvgis.columns.str.strip()

print(f"  Loaded  : {pvgis.shape[0]:,} rows × {pvgis.shape[1]} cols")
print(f"  Columns : {pvgis.columns.tolist()}")
print(f"  Dtypes  :\n{pvgis.dtypes}")
print(f"  Sample  :\n{pvgis.head(3).to_string()}")

# Parse PVGIS timestamp  "YYYYMMDD:HHMM"  →  datetime (UTC)
# Example: "20190101:0030"  →  2019-01-01 00:30:00 UTC
pvgis["datetime_raw"] = pd.to_datetime(
    pvgis["time"], format="%Y%m%d:%H%M", utc=True
)

# TIME ALIGNMENT:  PVGIS timestamps are at :30 (mid-hour representation).
# Subtract 30 minutes  →  :00  so they align with NASA's top-of-hour stamps.
# After subtraction:  20190101:0030 → 2019-01-01 00:00 UTC  (same hour as NASA 2019010100)
pvgis["datetime"] = pvgis["datetime_raw"] - pd.Timedelta(minutes=30)

print(f"\n  Timestamp range after alignment:")
print(f"    First : {pvgis['datetime'].iloc[0]}")
print(f"    Last  : {pvgis['datetime'].iloc[-1]}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — LOAD & PARSE  ─ NASA POWER 2019 + 2020
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 2 — Loading NASA POWER 2019 + 2020")
print("=" * 60)

nasa19 = pd.read_csv(NASA_2019)
nasa20 = pd.read_csv(NASA_2020)
nasa19.columns = nasa19.columns.str.strip()
nasa20.columns = nasa20.columns.str.strip()

print(f"  NASA 2019  : {nasa19.shape[0]:,} rows")
print(f"  NASA 2020  : {nasa20.shape[0]:,} rows")
print(f"  Columns    : {nasa19.columns.tolist()}")
print(f"  Sample     :\n{nasa19.head(3).to_string()}")

# Concatenate years
nasa = pd.concat([nasa19, nasa20], ignore_index=True)
print(f"  Combined   : {nasa.shape[0]:,} rows")

# Parse NASA timestamp  "YYYYMMDDHH"  → datetime (UTC) at top of hour
# Example: "2019010100"  →  2019-01-01 00:00:00 UTC
nasa["datetime"] = pd.to_datetime(
    nasa["time"].astype(str), format="%Y%m%d%H", utc=True
)

print(f"\n  NASA timestamp range:")
print(f"    First : {nasa['datetime'].iloc[0]}")
print(f"    Last  : {nasa['datetime'].iloc[-1]}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — LOAD & PARSE  ─ PVGIS TMY  (for WD10m)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 3 — Loading PVGIS TMY (WD10m source)")
print("=" * 60)

tmy = pd.read_csv(PVGIS_TMY)
tmy.columns = tmy.columns.str.strip()

print(f"  Loaded  : {tmy.shape[0]:,} rows × {tmy.shape[1]} cols")
print(f"  Columns : {tmy.columns.tolist()}")
print(f"  Sample  :\n{tmy.head(3).to_string()}")

# Parse TMY timestamp  "YYYYMMDD:HHMM"  at :00
tmy["datetime_tmy"] = pd.to_datetime(
    tmy["time(UTC)"].str.strip(), format="%Y%m%d:%H%M", utc=True
)

# TMY is a "typical year" — the year value is arbitrary.
# We join by (month, day, hour) pattern only.
tmy["_month"] = tmy["datetime_tmy"].dt.month
tmy["_day"]   = tmy["datetime_tmy"].dt.day
tmy["_hour"]  = tmy["datetime_tmy"].dt.hour

# Keep only what we need from TMY
tmy_wd = tmy[["_month", "_day", "_hour", "WD10m"]].copy()
# Drop any duplicate (month, day, hour) keys
tmy_wd = tmy_wd.drop_duplicates(subset=["_month", "_day", "_hour"])
print(f"  TMY WD10m keys  : {tmy_wd.shape[0]:,} unique (month, day, hour) slots")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SENTINEL & UNIT CONVERSIONS  (before merge)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 4 — Sentinel replacement & unit conversions")
print("=" * 60)

# --- PVGIS hourly ---
numeric_pvgis = ["P", "Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m", "Int"]
pvgis[numeric_pvgis] = pvgis[numeric_pvgis].apply(pd.to_numeric, errors="coerce")
pvgis = replace_sentinels(pvgis)

# Derived column: total irradiance on the inclined panel
pvgis["G_i"] = pvgis["Gb(i)"] + pvgis["Gd(i)"] + pvgis["Gr(i)"]

print(f"  PVGIS sentinels replaced — remaining NaN:\n{pvgis[numeric_pvgis].isnull().sum().to_dict()}")

# --- NASA ---
numeric_nasa = ["T2M", "RH2M", "WS10M", "PS", "ALLSKY_SFC_SW_DWN"]
nasa[numeric_nasa] = nasa[numeric_nasa].apply(pd.to_numeric, errors="coerce")
nasa = replace_sentinels(nasa)

# Unit conversion: NASA PS is in kPa  →  Pa  (×1000)
# Verify: typical value ~98 kPa  →  98,000 Pa
nasa["PS_Pa"] = nasa["PS"] * 1000.0
print(f"\n  NASA PS  (kPa) sample: {nasa['PS'].head(3).tolist()}")
print(f"  NASA PS  (Pa)  sample: {nasa['PS_Pa'].head(3).tolist()}")

# Note: ALLSKY_SFC_SW_DWN from NASA hourly endpoint is in W/m².
# Values ~0 at night, hundreds of W/m² during day confirms W/m² unit.
# No conversion needed.  We keep it as a quality cross-check column only.
print(f"\n  NASA ALLSKY sample (W/m², daytime hours 7-9):")
print(f"  {nasa.loc[nasa['datetime'].dt.hour.isin([7,8,9]), 'ALLSKY_SFC_SW_DWN'].head(5).tolist()}")

print(f"\n  NASA sentinels replaced — remaining NaN:\n{nasa[numeric_nasa].isnull().sum().to_dict()}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — MERGE PVGIS HOURLY  +  NASA  (primary merge)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 5 — Merging PVGIS hourly with NASA (left join on datetime)")
print("=" * 60)

# Select only the columns we want from NASA
nasa_merge = nasa[["datetime", "RH2M", "PS_Pa"]].copy()

# Left merge: PVGIS is primary (keeps all 17,544 PVGIS rows)
combined = pvgis.merge(nasa_merge, on="datetime", how="left")

print(f"  PVGIS rows   : {pvgis.shape[0]:,}")
print(f"  NASA rows    : {nasa.shape[0]:,}")
print(f"  After merge  : {combined.shape[0]:,} rows  (should equal PVGIS rows)")
print(f"  RH2M NaN     : {combined['RH2M'].isnull().sum()}")
print(f"  PS_Pa NaN    : {combined['PS_Pa'].isnull().sum()}")

if combined["RH2M"].isnull().sum() > 0:
    print("\n  ⚠  Some PVGIS rows did not find a NASA match.")
    print("     Unmatched PVGIS datetimes (first 5):")
    mask = combined["RH2M"].isnull()
    print(f"  {combined.loc[mask, 'datetime'].head(5).tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 6 — MERGE PVGIS TMY  (adds WD10m by seasonal pattern)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 6 — Merging TMY WD10m by (month, day, hour) pattern")
print("=" * 60)

combined["_month"] = combined["datetime"].dt.month
combined["_day"]   = combined["datetime"].dt.day
combined["_hour"]  = combined["datetime"].dt.hour

combined = combined.merge(tmy_wd, on=["_month", "_day", "_hour"], how="left")

print(f"  WD10m NaN after merge: {combined['WD10m'].isnull().sum()}")
# Drop helper keys
combined.drop(columns=["_month", "_day", "_hour"], inplace=True)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 7 — MISSING VALUE HANDLING
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 7 — Missing value handling")
print("=" * 60)

feature_cols = ["P", "G_i", "Gb(i)", "Gd(i)", "Gr(i)", "H_sun",
                "T2m", "WS10m", "RH2M", "PS_Pa", "WD10m"]

nan_before = combined[feature_cols].isnull().sum()
print("  NaN counts BEFORE filling:\n", nan_before[nan_before > 0].to_string())

# Strategy:
#   • Physical zeros at night are NOT missing — they are valid zero-irradiance.
#   • NaN in meteorological columns: interpolate linearly (max 3-hour gap).
#   • NaN in irradiance / power during daytime only: flag — do not guess.
#   • Remaining NaN after interpolation: forward-fill then back-fill.

met_cols = ["T2m", "WS10m", "RH2M", "PS_Pa", "WD10m"]

# Pandas requires a DatetimeIndex for method="time" interpolation.
combined_idx = combined.set_index("datetime")
for col in met_cols:
    combined_idx[col] = combined_idx[col].interpolate(
        method="time", limit=3, limit_direction="both"
    )
    combined_idx[col] = combined_idx[col].ffill().bfill()

combined[met_cols] = combined_idx[met_cols].values

nan_after = combined[feature_cols].isnull().sum()
print("\n  NaN counts AFTER filling:\n", nan_after[nan_after > 0].to_string() or "  None remaining ✓")

# Int==1 means the record was interpolated by PVGIS itself → flag, but keep.
# Add a 'data_quality' column: 0 = clean, 1 = PVGIS-interpolated, 2 = gap-filled
combined["data_quality"] = 0
combined.loc[combined["Int"] == 1, "data_quality"] = 1
print(f"\n  PVGIS-interpolated rows (Int=1)  : {(combined['Int'] == 1).sum():,}")
print(f"  Clean rows (Int=0)               : {(combined['Int'] == 0).sum():,}")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 8 — FINAL COLUMN SELECTION, RENAME & VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 8 — Final column selection & validation")
print("=" * 60)

final = combined[[
    "datetime",      # UTC timestamp (hourly, aligned to :00)
    "P",             # TARGET: PV power output (W) for 1 kWp system
    "G_i",           # Total global irradiance on inclined plane (W/m²)
    "Gb(i)",         # Beam irradiance on inclined plane (W/m²)
    "Gd(i)",         # Diffuse irradiance on inclined plane (W/m²)
    "Gr(i)",         # Reflected irradiance on inclined plane (W/m²)
    "H_sun",         # Sun elevation angle (degrees)
    "T2m",           # Air temperature at 2 m (°C)
    "WS10m",         # Wind speed at 10 m (m/s)
    "RH2M",          # Relative humidity at 2 m (%)   ← from NASA
    "PS_Pa",         # Surface pressure (Pa)           ← from NASA, converted
    "WD10m",         # Wind direction (°)              ← from TMY seasonal pattern
    "data_quality",  # 0=clean, 1=PVGIS-interpolated
]].copy()

# Rename to clean, consistent names
final.rename(columns={
    "Gb(i)"  : "Gb_i",
    "Gd(i)"  : "Gd_i",
    "Gr(i)"  : "Gr_i",
    "RH2M"   : "RH",
    "PS_Pa"  : "SP",
    "WS10m"  : "WS10m",    # already correct
}, inplace=True)

print(f"  Final columns  : {final.columns.tolist()}")
print(f"  Final shape    : {final.shape}")
print(f"  Date range     : {final['datetime'].iloc[0]}  →  {final['datetime'].iloc[-1]}")

# --- Value range checks ---
checks = {
    "P"      : (0,    1100),   # 0–1100 W for a 1 kWp system
    "G_i"    : (0,    1600),   # W/m²
    "H_sun"  : (-90,  90),     # degrees
    "T2m"    : (-10,  55),     # °C  (Rajasthan range)
    "WS10m"  : (0,    30),     # m/s
    "RH"     : (0,    100),    # %
    "SP"     : (80000, 105000), # Pa
    "WD10m"  : (0,    360),    # degrees
}

print("\n  Physical range validation:")
all_ok = True
for col, (lo, hi) in checks.items():
    n_lo = (final[col] < lo).sum()
    n_hi = (final[col] > hi).sum()
    nans = final[col].isnull().sum()
    status = "✓" if (n_lo == 0 and n_hi == 0 and nans == 0) else "⚠ "
    if status != "✓":
        all_ok = False
    print(f"    {status}  {col:10s}  below {lo}: {n_lo:4d}  above {hi}: {n_hi:4d}  NaN: {nans:4d}")

if all_ok:
    print("\n  All range checks passed ✓")

# Duplicate timestamp check
dup = final["datetime"].duplicated().sum()
print(f"\n  Duplicate timestamps : {dup}")

# Expected row count: 2019 (8760 h) + 2020 (8784 h) = 17,544
expected = 17544
print(f"  Row count : {final.shape[0]:,}  (expected {expected:,})")
if final.shape[0] != expected:
    print(f"  ⚠  Row count mismatch!")


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 9 — EXPORT
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STAGE 9 — Exporting processed dataset")
print("=" * 60)

# Convert tz-aware datetime to ISO string for CSV portability
final["datetime"] = final["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

final.to_csv(OUTPUT, index=False)
print(f"  ✓  Saved: {OUTPUT}")
print(f"  Rows   : {final.shape[0]:,}")
print(f"  Cols   : {final.shape[1]}")

# Summary statistics
print("\n  Descriptive statistics (numeric columns):")
desc = final.drop(columns=["datetime"]).describe().round(3)
print(desc.to_string())

# ──────────────────────────────────────────────────────────────────────────────
# CLEANING REPORT
# ──────────────────────────────────────────────────────────────────────────────
report_lines = [
    "Solar PV Data Cleaning Report",
    "=" * 60,
    f"Output file   : {OUTPUT.name}",
    f"Total rows    : {final.shape[0]:,}",
    f"Total columns : {final.shape[1]}",
    "",
    "Source file contributions:",
    f"  PVGIS hourly 2019-2020  → P, Gb_i, Gd_i, Gr_i, G_i, H_sun, T2m, WS10m, Int",
    f"  NASA POWER 2019-2020    → RH (from RH2M), SP (from PS×1000)",
    f"  PVGIS TMY               → WD10m (seasonal pattern by month-day-hour)",
    "",
    "Timestamp alignment:",
    "  PVGIS  :30 timestamps − 30 min → :00 (hour-start convention)",
    "  NASA   :00 timestamps → used as-is",
    "  Merge key: exact datetime match (left join, PVGIS primary)",
    "",
    "Unit conversions applied:",
    "  NASA PS  (kPa) × 1000 → SP (Pa)",
    "  NASA ALLSKY_SFC_SW_DWN kept for reference only (W/m², no conversion)",
    "",
    f"Data quality flags:",
    f"  data_quality=0  (clean)               : {(final['data_quality']==0).sum():,}",
    f"  data_quality=1  (PVGIS-interpolated)  : {(final['data_quality']==1).sum():,}",
    "",
    "Column descriptions:",
    "  datetime    UTC timestamp (hourly, :00)",
    "  P           TARGET — PV power output (W) for 1 kWp system @ 30° tilt south",
    "  G_i         Total global irradiance on inclined panel = Gb_i+Gd_i+Gr_i (W/m²)",
    "  Gb_i        Beam irradiance on inclined plane (W/m²)",
    "  Gd_i        Diffuse irradiance on inclined plane (W/m²)",
    "  Gr_i        Ground-reflected irradiance on inclined plane (W/m²)",
    "  H_sun       Sun elevation angle above horizon (°), negative=below horizon",
    "  T2m         Air temperature at 2 m height (°C)",
    "  WS10m       Wind speed at 10 m height (m/s)",
    "  RH          Relative humidity at 2 m (%)  [NASA RH2M]",
    "  SP          Surface atmospheric pressure (Pa)  [NASA PS × 1000]",
    "  WD10m       Wind direction at 10 m (°)  [TMY seasonal pattern]",
    "  data_quality 0=clean, 1=PVGIS-interpolated (Int flag)",
]

with open(REPORT, "w") as f:
    f.write("\n".join(report_lines))

print(f"\n  ✓  Report saved: {REPORT.name}")
print("\nPipeline complete.")
