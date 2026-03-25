"""
Radekhiv Solar PV - Preprocessing Pipeline for Dual Stream CNN-LSTM
=====================================================================
Dataset  : ../data/Shakhovska_Cleaned.csv  (output of Radekhiv_dataset_cleaning.py)
Location : Radekhiv, Western Ukraine  (lat=50.2797, lon=24.6369, alt=231 m)
Target   : generation  — hourly PV power output (kW)
Date range: June 2022 – February 2024  (~13,052 rows after cleaning)

Pipeline mirrors the PVGIS scaling.py structure so the same dual_stream.py
model can be used on both datasets with no architecture changes.

Feature Groups → Scaler Mapping
---------------------------------
  SPATIAL   : solarradiation, solarenergy, cloudcover, visibility
              → MinMaxScaler [0, 1]   (irradiance-type bounded quantities)

  TEMPORAL  : temp, humidity, windspeed, sealevelpressure
              → StandardScaler        (meteorological, unbounded range)

  ANGLE     : sunheight, winddir
              → MinMaxScaler [-1, 1]  (angle-type features)

  CYCLIC    : hour_sin/cos, month_sin/cos, doy_sin/cos
              → No scaling (already in [-1, 1])

  FLAGS     : is_daytime, is_peak_hours
              → No scaling (binary 0/1)

  TARGET    : generation
              → MinMaxScaler [0, 1]

Dropped before modeling:
  - feelslike   (near-duplicate of temp, adds no info)
  - dew         (collinear with humidity/temp)
  - precip, precipprob, snow, snowdepth, windgust, severerisk
                (very low correlation with generation; add noise)
  - uvindex     (collinear with solarradiation)
  - conditions, icon, season  (categorical; temporal features cover seasonality)

Stages
------
  1. Load & validate
  2. Feature engineering  (cyclic time + flags)
  3. Feature selection    (drop low-value columns)
  4. Daytime filter       (solarradiation > 0)
  5. Temporal split       (70 / 20 / 10)
  6. Scaling              (fit on train only)
  7. Sequence generation  (lookback=2, horizon=1)
  8. Quality validation
  9. Save

Author : [Your Name]
Date   : March 2026
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """All tunable parameters in one place."""

    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_PATH  = Path("../data/Shakhovska_Cleaned.csv")  # adjust if needed
    OUTPUT_DIR = Path("radekhiv_preprocessed_data")

    # ── Sequence parameters (match PVGIS pipeline & dual_stream.py) ────────
    LOOKBACK_WINDOW  = 2   # hours of history fed into each sample
    FORECAST_HORIZON = 1   # hours ahead to predict

    # ── Temporal split ratios ───────────────────────────────────────────────
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.20
    TEST_RATIO  = 0.10     # remainder

    # ── Feature groups (used by scalers + sequence builder) ────────────────
    # Irradiance-type: solar quantities + visibility + cloudcover
    SPATIAL_FEATURES  = ["solarradiation", "solarenergy", "cloudcover", "visibility"]

    # Meteorological: temperature, humidity, wind speed, pressure
    TEMPORAL_FEATURES = ["temp", "humidity", "windspeed", "sealevelpressure"]

    # Angle-type: sun elevation + wind direction
    ANGLE_FEATURES    = ["sunheight", "winddir"]

    # Target
    TARGET = "generation"

    # ── Filtering ──────────────────────────────────────────────────────────
    USE_DAYTIME_ONLY = True   # solarradiation > 0

    # ── Reproducibility ────────────────────────────────────────────────────
    RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def load_and_validate(config: Config) -> pd.DataFrame:
    """Load ../data/Shakhovska_Cleaned.csv and run basic sanity checks."""

    print("\n" + "=" * 70)
    print("STAGE 1: LOAD & VALIDATE")
    print("=" * 70)

    df = pd.read_csv(config.DATA_PATH)

    # Datetime parsing — the cleaned file has a plain 'Datetime' column
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    print(f"✓ Loaded : {len(df):,} rows × {df.shape[1]} cols")
    print(f"  Date range : {df['Datetime'].iloc[0]}  →  {df['Datetime'].iloc[-1]}")

    # ── Confirm expected columns are present ───────────────────────────────
    required = (
        [config.TARGET, "Datetime", "sunheight"]
        + config.SPATIAL_FEATURES
        + config.TEMPORAL_FEATURES
        + config.ANGLE_FEATURES
    )
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    print(f"✓ All required columns present")

    # ── Basic value checks ─────────────────────────────────────────────────
    checks = {
        "generation"      : (0,   None),
        "solarradiation"  : (0,   None),
        "temp"            : (-40, 50),
        "humidity"        : (0,   100),
        "windspeed"       : (0,   None),
        "sealevelpressure": (900, 1100),
    }
    print("\n  Physical range validation:")
    for col, (lo, hi) in checks.items():
        if col not in df.columns:
            continue
        n_lo = (df[col] < lo).sum() if lo is not None else 0
        n_hi = (df[col] > hi).sum() if hi is not None else 0
        nans = df[col].isnull().sum()
        ok   = (n_lo == 0 and n_hi == 0 and nans == 0)
        flag = "✓" if ok else "⚠"
        print(f"    {flag}  {col:20s}  below {str(lo):>5}: {n_lo:4d}  "
              f"above {str(hi):>5}: {n_hi:4d}  NaN: {nans:4d}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic time encodings and binary flags.

    New columns
    -----------
    hour_sin, hour_cos   — 24-hour cycle
    month_sin, month_cos — 12-month cycle
    doy_sin,  doy_cos    — 365-day cycle
    is_daytime           — 1 if sunheight > 0  (sun above horizon)
    is_peak_hours        — 1 if UTC hour in [6, 12]
                           (covers solar noon band for Ukraine UTC+2/+3)
    """
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()

    hour  = df["Datetime"].dt.hour
    month = df["Datetime"].dt.month
    doy   = df["Datetime"].dt.dayofyear

    # Cyclic encodings
    df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * doy   / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * doy   / 365)

    # Binary flags
    df["is_daytime"]    = (df["sunheight"] > 0).astype(int)
    df["is_peak_hours"] = (hour.between(6, 12)).astype(int)

    print("✓ Added 8 new features:")
    print("    Cyclic  : hour_sin/cos, month_sin/cos, doy_sin/cos")
    print("    Binary  : is_daytime (sunheight > 0), is_peak_hours (UTC 6-12)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Keep only columns needed for the dual-stream model.
    Drops redundant, collinear, and categorical columns.

    Dropped columns and reasons
    ---------------------------
    feelslike       — near-duplicate of temp (r > 0.99)
    dew             — collinear with humidity + temp
    precip          — near-zero most of the time, low signal
    precipprob      — same issue
    snow / snowdepth— mostly zero in this dataset
    windgust        — collinear with windspeed
    uvindex         — collinear with solarradiation
    severerisk      — near-constant (mostly 0)
    conditions      — categorical; covered by cyclic + flag features
    icon            — categorical duplicate of conditions
    season          — categorical; covered by month_sin/cos + doy_sin/cos
    """
    print("\n" + "=" * 70)
    print("STAGE 3: FEATURE SELECTION")
    print("=" * 70)

    cyclic_flags = [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "doy_sin",  "doy_cos",
        "is_daytime", "is_peak_hours",
    ]

    keep = (
        ["Datetime", config.TARGET]
        + config.SPATIAL_FEATURES
        + config.TEMPORAL_FEATURES
        + config.ANGLE_FEATURES
        + cyclic_flags
    )

    dropped = [c for c in df.columns if c not in keep]
    df = df[keep].copy()

    print(f"✓ Kept {len(keep)} columns, dropped {len(dropped)}")
    print(f"  Dropped: {dropped}")
    print(f"\n  Final column list:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: DAYTIME FILTER
# ══════════════════════════════════════════════════════════════════════════════

def filter_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Filter to daytime rows only (solarradiation > 0)."""

    print("\n" + "=" * 70)
    print("STAGE 4: DAYTIME FILTER")
    print("=" * 70)

    before = len(df)
    if config.USE_DAYTIME_ONLY:
        df = df[df["solarradiation"] > 0].reset_index(drop=True)
        print(f"✓ solarradiation > 0: {before:,} → {len(df):,} rows "
              f"({before - len(df):,} night rows removed)")
    else:
        print("  Daytime filter skipped (USE_DAYTIME_ONLY = False)")

    nans = df.isnull().sum()
    if nans.sum() > 0:
        print(f"\n⚠ Remaining NaN values:\n{nans[nans > 0]}")
    else:
        print("✓ No missing values in filtered dataset")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: TEMPORAL TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(df: pd.DataFrame, config: Config):
    """Chronological 70/20/10 split — no shuffling."""

    print("\n" + "=" * 70)
    print("STAGE 5: TEMPORAL TRAIN / VAL / TEST SPLIT")
    print("=" * 70)

    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end   = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"  {name:5s}: {len(split):,} rows ({len(split)/n*100:.1f}%)  "
              f"[{split['Datetime'].iloc[0].date()}  →  {split['Datetime'].iloc[-1].date()}]")

    return df_train, df_val, df_test


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6: SCALING
# ══════════════════════════════════════════════════════════════════════════════

def create_scalers(df_train: pd.DataFrame, config: Config) -> dict:
    """
    Fit all scalers on training data only to prevent data leakage.

    spatial  → MinMaxScaler [0, 1]   — bounded, non-negative quantities
    temporal → StandardScaler         — can be negative (temp, pressure)
    angle    → MinMaxScaler [-1, 1]   — signed angle-type features
    target   → MinMaxScaler [0, 1]    — generation output
    """
    print("\n" + "=" * 70)
    print("STAGE 6: FEATURE SCALING")
    print("=" * 70)

    scalers = {}

    scalers["spatial"] = MinMaxScaler(feature_range=(0, 1))
    scalers["spatial"].fit(df_train[config.SPATIAL_FEATURES])
    print(f"✓ spatial  scaler (MinMax [0,1])   → {config.SPATIAL_FEATURES}")

    scalers["temporal"] = StandardScaler()
    scalers["temporal"].fit(df_train[config.TEMPORAL_FEATURES])
    print(f"✓ temporal scaler (StandardScaler) → {config.TEMPORAL_FEATURES}")

    scalers["angle"] = MinMaxScaler(feature_range=(-1, 1))
    scalers["angle"].fit(df_train[config.ANGLE_FEATURES])
    print(f"✓ angle    scaler (MinMax [-1,1])  → {config.ANGLE_FEATURES}")

    scalers["target"] = MinMaxScaler(feature_range=(0, 1))
    scalers["target"].fit(df_train[[config.TARGET]])
    print(f"✓ target   scaler (MinMax [0,1])   → {config.TARGET}")

    print("✓ Cyclic + flag features: no scaling needed")

    # Print scaler statistics for documentation
    print("\n  Scaler statistics (training data):")
    print("    Spatial  — data_min:", np.round(scalers["spatial"].data_min_, 2).tolist())
    print("    Spatial  — data_max:", np.round(scalers["spatial"].data_max_, 2).tolist())
    print("    Temporal — mean    :", np.round(scalers["temporal"].mean_, 2).tolist())
    print("    Temporal — std     :", np.round(scalers["temporal"].scale_, 2).tolist())
    print("    Target   — data_min:", np.round(scalers["target"].data_min_, 2).tolist())
    print("    Target   — data_max:", np.round(scalers["target"].data_max_, 2).tolist())

    return scalers


def apply_scaling(df: pd.DataFrame, scalers: dict, config: Config) -> pd.DataFrame:
    """Apply pre-fitted scalers to one data split."""
    df_s = df.copy()
    df_s[config.SPATIAL_FEATURES]  = scalers["spatial"].transform(df[config.SPATIAL_FEATURES])
    df_s[config.TEMPORAL_FEATURES] = scalers["temporal"].transform(df[config.TEMPORAL_FEATURES])
    df_s[config.ANGLE_FEATURES]    = scalers["angle"].transform(df[config.ANGLE_FEATURES])
    df_s[[config.TARGET]]          = scalers["target"].transform(df[[config.TARGET]])
    return df_s


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7: SEQUENCE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(data: np.ndarray, target: np.ndarray,
                     lookback: int = 2, horizon: int = 1):
    """
    Slide a window to produce (X, y) pairs.

      X[i] = data[i : i+lookback]           shape (lookback, n_features)
      y[i] = target[i + lookback + horizon - 1]
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i])
        y.append(target[i + horizon - 1])
    return np.array(X), np.array(y)


def generate_sequences(df_train, df_val, df_test,
                        config: Config, feature_columns: list) -> dict:
    """Build and return sequences for all three splits."""

    print("\n" + "=" * 70)
    print("STAGE 7: SEQUENCE GENERATION")
    print("=" * 70)
    print(f"  Lookback  : {config.LOOKBACK_WINDOW} h")
    print(f"  Horizon   : {config.FORECAST_HORIZON} h")
    print(f"  Features  : {len(feature_columns)}")

    seq = {}
    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X, y = create_sequences(
            split[feature_columns].values,
            split[config.TARGET].values,
            config.LOOKBACK_WINDOW,
            config.FORECAST_HORIZON,
        )
        seq[f"X_{name}"] = X
        seq[f"y_{name}"] = y
        print(f"  ✓ {name:5s}  X={X.shape}  y={y.shape}")

    assert seq["X_train"].shape[1] == config.LOOKBACK_WINDOW, "Lookback mismatch"
    assert seq["X_train"].shape[2] == len(feature_columns), "Feature count mismatch"
    print("✓ Shape assertions passed")

    return seq


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8: QUALITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_data(seq: dict) -> bool:
    """Six checks on the generated arrays."""

    print("\n" + "=" * 70)
    print("STAGE 8: DATA QUALITY VALIDATION")
    print("=" * 70)

    checks = {
        "No NaN values"                  : all(np.isnan(seq[k]).sum() == 0 for k in seq),
        "No Inf values"                  : all(np.isinf(seq[k]).sum() == 0 for k in seq),
        "Target in [0, ~1.5]"            : all(
            seq[f"y_{s}"].min() >= 0 and seq[f"y_{s}"].max() <= 1.5
            for s in ["train", "val", "test"]
        ),
        "Train larger than val/test"     : (
            len(seq["X_train"]) > len(seq["X_val"]) and
            len(seq["X_train"]) > len(seq["X_test"])
        ),
        "Consistent sequence shapes"     : (
            seq["X_train"].shape[1:] ==
            seq["X_val"].shape[1:]   ==
            seq["X_test"].shape[1:]
        ),
        "Sufficient training samples (>500)": len(seq["X_train"]) > 500,
    }

    passed = sum(checks.values())
    for msg, ok in checks.items():
        print(f"  {'✓' if ok else '✗'}  {msg}")

    print(f"\n{'✓' if passed == len(checks) else '⚠'} {passed}/{len(checks)} checks passed")
    return passed == len(checks)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9: SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_preprocessed_data(seq: dict, scalers: dict,
                            feature_columns: list, config: Config) -> None:
    """Save arrays, scalers, and metadata."""

    print("\n" + "=" * 70)
    print("STAGE 9: SAVING PREPROCESSED DATA")
    print("=" * 70)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Numpy arrays
    for key, arr in seq.items():
        np.save(config.OUTPUT_DIR / f"{key}.npy", arr)
    print(f"✓ Saved {len(seq)} numpy arrays  (X/y train/val/test)")

    # Scalers
    with open(config.OUTPUT_DIR / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    print("✓ Saved scalers.pkl")

    # Metadata
    metadata = {
        "dataset"          : "Radekhiv, Western Ukraine",
        "location"         : {"lat": 50.2797, "lon": 24.6369, "alt_m": 231},
        "feature_columns"  : feature_columns,
        "lookback_window"  : config.LOOKBACK_WINDOW,
        "forecast_horizon" : config.FORECAST_HORIZON,
        "train_size"       : int(len(seq["X_train"])),
        "val_size"         : int(len(seq["X_val"])),
        "test_size"        : int(len(seq["X_test"])),
        "n_features"       : len(feature_columns),
        "target_column"    : config.TARGET,
        "daytime_only"     : config.USE_DAYTIME_ONLY,
        "preprocessing_date": pd.Timestamp.now().isoformat(),
    }
    with open(config.OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Saved metadata.json")

    # Feature info
    feature_info = {
        "spatial_features" : config.SPATIAL_FEATURES,
        "temporal_features": config.TEMPORAL_FEATURES,
        "angle_features"   : config.ANGLE_FEATURES,
        "cyclic_features"  : ["hour_sin", "hour_cos",
                               "month_sin", "month_cos",
                               "doy_sin",  "doy_cos"],
        "flag_features"    : ["is_daytime", "is_peak_hours"],
        "target"           : config.TARGET,
    }
    with open(config.OUTPUT_DIR / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    print("✓ Saved feature_info.json")

    print(f"\n  Files in {config.OUTPUT_DIR}/")
    for p in sorted(config.OUTPUT_DIR.iterdir()):
        print(f"    {p.name:30s}  ({p.stat().st_size / 1024:.1f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print(" " * 10 + "RADEKHIV  CNN-LSTM  PREPROCESSING  PIPELINE")
    print("=" * 70)
    print(f"  Input  : {Config.DATA_PATH}")
    print(f"  Output : {Config.OUTPUT_DIR}")

    df = load_and_validate(Config)
    df = add_temporal_features(df)
    df = select_features(df, Config)
    df = filter_data(df, Config)

    df_train, df_val, df_test = temporal_split(df, Config)

    scalers     = create_scalers(df_train, Config)
    df_train_sc = apply_scaling(df_train, scalers, Config)
    df_val_sc   = apply_scaling(df_val,   scalers, Config)
    df_test_sc  = apply_scaling(df_test,  scalers, Config)
    print("✓ Scaling applied to all splits")

    # Feature column order — must remain stable for dual_stream.py
    feature_columns = (
        Config.SPATIAL_FEATURES    +   # 4: solarradiation, solarenergy, cloudcover, visibility
        Config.TEMPORAL_FEATURES   +   # 4: temp, humidity, windspeed, sealevelpressure
        Config.ANGLE_FEATURES      +   # 2: sunheight, winddir
        ["hour_sin",  "hour_cos",      # 6: cyclic time
         "month_sin", "month_cos",
         "doy_sin",   "doy_cos"]   +
        ["is_daytime", "is_peak_hours"]  # 2: binary flags
    )
    # Total: 18 features — same count as PVGIS pipeline

    print(f"\n  Feature columns ({len(feature_columns)} total):")
    for i, col in enumerate(feature_columns, 1):
        print(f"    {i:2d}. {col}")

    seq = generate_sequences(df_train_sc, df_val_sc, df_test_sc,
                              Config, feature_columns)

    ok = validate_data(seq)

    if ok:
        save_preprocessed_data(seq, scalers, feature_columns, Config)
    else:
        print("\n⚠ Validation failed — output NOT saved. Fix issues above.")
        return

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print("\nNext steps — load into dual_stream.py:")
    print(f"  X_train = np.load('{Config.OUTPUT_DIR}/X_train.npy')")
    print(f"  y_train = np.load('{Config.OUTPUT_DIR}/y_train.npy')")
    print(f"  # input_shape = (X_train.shape[1], X_train.shape[2])  → (2, 18)")
    print(f"  scalers = pickle.load(open('{Config.OUTPUT_DIR}/scalers.pkl','rb'))")
    print("\nTo invert predictions back to kW:")
    print("  y_kw = scalers['target'].inverse_transform(y_pred.reshape(-1, 1))")
    print("=" * 70)


if __name__ == "__main__":
    main()