"""
Radekhiv Solar PV — Dual-Stream CNN-LSTM Pipeline
===================================================
Location : Radekhiv, Western Ukraine  (lat=50.2797°N, lon=24.6369°E, alt=231 m)
Source   : Shakhovska et al. (2024) — figshare dataset
Target   : generation  (kWh or W, hourly PV power output)

Pipeline stages
---------------
  1. Load & validate the cleaned Radekhiv CSV
  2. Feature engineering (cyclic time, solar position flags, weather interactions)
  3. Data filtering  (daytime-only via solarradiation > 0)
  4. Temporal train / val / test split  (70 / 20 / 10)
  5. Feature scaling  (MinMax for irradiance, Standard for met, MinMax[-1,1] for angles)
  6. Sequence generation  (lookback=24 h, horizon=1 h)
  7. Quality validation
  8. Save preprocessed arrays + scalers + metadata
  9. Build & train DSCLANet (Dual-Stream CNN-LSTM with Attention)
 10. Evaluate & save metrics

Key differences from the PVGIS/Rajasthan pipeline
--------------------------------------------------
- Target column     : 'generation'  (not 'P')
- Irradiance proxy  : 'solarradiation' + 'solarenergy'   (no beam/diffuse decomposition)
- Met features      : 'temp', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover'
- Sun angle         : 'sunheight'  (pvlib apparent_elevation, already in cleaned file)
- Extra features    : 'snow', 'snowdepth', 'visibility', 'windgust', 'dew' (Ukraine climate)
- Lookback window   : 24 h  (captures full diurnal cycle — important for temperate climate)
- Daytime filter    : solarradiation > 0  (consistent with Radekhiv_dataset_cleaning.py)
- No UTC shift needed (Visual Crossing timestamps are already local or consistent)
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """All tunable parameters in one place."""

    # ── Paths ──────────────────────────────────────────────────────────────
    # Update DATA_PATH to wherever Shakhovska_Cleaned.csv lives on your machine
    DATA_PATH  = Path("../data/Shakhovska_Cleaned.csv")
    OUTPUT_DIR = Path("radekhiv_preprocessed")
    MODEL_DIR  = Path("radekhiv_models")

    # ── Sequence parameters ────────────────────────────────────────────────
    LOOKBACK_WINDOW  = 24   # 24 hours of history captures a full diurnal cycle
    FORECAST_HORIZON = 1    # predict 1 hour ahead

    # ── Temporal split ─────────────────────────────────────────────────────
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.20
    TEST_RATIO  = 0.10

    # ── Feature groups ─────────────────────────────────────────────────────
    # Irradiance / solar features  → MinMaxScaler [0, 1]
    SOLAR_FEATURES = ["solarradiation", "solarenergy", "uvindex"]

    # Meteorological features  → StandardScaler
    MET_FEATURES = ["temp", "feelslike", "dew", "humidity",
                    "windspeed", "windgust", "sealevelpressure",
                    "cloudcover", "visibility"]

    # Precipitation / snow features  → MinMaxScaler [0, 1]
    PRECIP_FEATURES = ["precip", "precipprob", "snow", "snowdepth"]

    # Angle feature  → MinMaxScaler [-1, 1]
    ANGLE_FEATURES = ["sunheight", "winddir"]

    # Target
    TARGET = "generation"

    # ── Filtering ──────────────────────────────────────────────────────────
    USE_DAYTIME_ONLY = True   # solarradiation > 0

    # ── Model hyperparameters ──────────────────────────────────────────────
    CNN_FILTERS    = [32, 64, 128]
    CNN_KERNELS    = [5, 3, 1]
    LSTM_UNITS     = [100, 100]
    DENSE_UNITS    = [64, 32, 12]
    DROPOUT_RATE   = 0.2
    LEARNING_RATE  = 0.001
    BATCH_SIZE     = 32
    MAX_EPOCHS     = 100
    PATIENCE       = 15

    # ── Reproducibility ────────────────────────────────────────────────────
    RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def load_and_validate(config: Config) -> pd.DataFrame:
    """Load cleaned Radekhiv CSV and run basic sanity checks."""
    print("\n" + "=" * 70)
    print("STAGE 1: LOAD & VALIDATE")
    print("=" * 70)

    df = pd.read_csv(config.DATA_PATH)

    # Parse datetime  ─  column is named 'Datetime' in the cleaned file
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    print(f"  Loaded   : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Date range: {df['Datetime'].iloc[0]}  →  {df['Datetime'].iloc[-1]}")
    print(f"  Columns  : {df.columns.tolist()}")

    # Required columns check
    all_features = (
        config.SOLAR_FEATURES + config.MET_FEATURES +
        config.PRECIP_FEATURES + config.ANGLE_FEATURES +
        [config.TARGET, "Datetime"]
    )
    missing_cols = [c for c in all_features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")
    print(f"  ✓ All required columns present")

    # Missing value report
    nan_counts = df[all_features].isnull().sum()
    if nan_counts.sum() > 0:
        print(f"  ⚠  NaN values found:\n{nan_counts[nan_counts > 0].to_string()}")
    else:
        print(f"  ✓ No missing values")

    # Value range checks
    print(f"\n  Physical range checks:")
    checks = {
        "generation"     : (-0.5,  500),
        "solarradiation" : (0,     1500),
        "temp"           : (-30,   45),
        "humidity"       : (0,     100),
        "sealevelpressure": (950,  1060),
        "sunheight"      : (-90,   90),
    }
    for col, (lo, hi) in checks.items():
        if col not in df.columns:
            continue
        n_lo = (df[col] < lo).sum()
        n_hi = (df[col] > hi).sum()
        status = "✓" if (n_lo == 0 and n_hi == 0) else "⚠ "
        print(f"    {status}  {col:20s}  below {lo}: {n_lo}  above {hi}: {n_hi}")

    print(f"\n  Target stats (generation):")
    print(f"    min={df['generation'].min():.2f}  max={df['generation'].max():.2f}"
          f"  mean={df['generation'].mean():.2f}  std={df['generation'].std():.2f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic time encoding, interaction features, and binary flags.

    New features
    ────────────
    Cyclic time   : hour_sin/cos, month_sin/cos, doy_sin/cos
    Binary flags  : is_daytime, is_peak_hours (peak solar 8–16 local ≈ Ukraine summer)
    Interactions  : cloud_rad (cloudcover × solarradiation)  — cloud attenuation proxy
                    temp_rad  (temp × solarradiation)         — temperature efficiency proxy
    """
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()

    # Extract time components
    df["hour"]       = df["Datetime"].dt.hour
    df["month"]      = df["Datetime"].dt.month
    df["day_of_year"]= df["Datetime"].dt.dayofyear

    # ── Cyclic encoding ────────────────────────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]       / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]       / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]      / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]      / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"]/ 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"]/ 365)

    # ── Binary flags ───────────────────────────────────────────────────────
    df["is_daytime"]    = (df["solarradiation"] > 0).astype(int)
    # Peak solar window for Western Ukraine (approx. 08:00–16:00 local)
    df["is_peak_hours"] = ((df["hour"] >= 8) & (df["hour"] <= 16)).astype(int)

    # ── Interaction features ───────────────────────────────────────────────
    # Normalise before multiplying so scale doesn't dominate
    max_cloud = df["cloudcover"].max() if df["cloudcover"].max() > 0 else 1.0
    max_rad   = df["solarradiation"].max() if df["solarradiation"].max() > 0 else 1.0
    df["cloud_rad"] = (df["cloudcover"] / max_cloud) * (df["solarradiation"] / max_rad)
    df["temp_rad"]  = df["temp"] * (df["solarradiation"] / max_rad)

    # Drop intermediate helper columns
    df = df.drop(columns=["hour", "month", "day_of_year"])

    new_feats = ["hour_sin","hour_cos","month_sin","month_cos","doy_sin","doy_cos",
                 "is_daytime","is_peak_hours","cloud_rad","temp_rad"]
    print(f"  ✓ Added {len(new_feats)} features: {new_feats}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — DATA FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_data(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Filter to daytime rows (solarradiation > 0)."""
    print("\n" + "=" * 70)
    print("STAGE 3: DATA FILTERING")
    print("=" * 70)

    n_before = len(df)

    if config.USE_DAYTIME_ONLY:
        df = df[df["solarradiation"] > 0].copy()
        print(f"  Daytime filter (solarradiation > 0): "
              f"{n_before:,} → {len(df):,} rows  "
              f"({n_before - len(df):,} night rows removed)")

    df = df.reset_index(drop=True)

    # Final NaN check after filtering
    nan_total = df.isnull().sum().sum()
    print(f"  ✓ Remaining NaN values: {nan_total}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — TEMPORAL TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(df: pd.DataFrame, config: Config):
    """Split temporally to prevent data leakage."""
    print("\n" + "=" * 70)
    print("STAGE 4: TEMPORAL TRAIN / VAL / TEST SPLIT")
    print("=" * 70)

    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end   = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"  {name:5s}: {len(split):,} rows  "
              f"({split['Datetime'].iloc[0].date()} → {split['Datetime'].iloc[-1].date()})")

    return df_train, df_val, df_test


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════════════

def create_and_fit_scalers(df_train: pd.DataFrame, config: Config) -> dict:
    """Fit all scalers on training data only."""
    print("\n" + "=" * 70)
    print("STAGE 5: FEATURE SCALING  (fit on train only)")
    print("=" * 70)

    scalers = {}

    # Solar / irradiance  → [0, 1]
    scalers["solar"] = MinMaxScaler(feature_range=(0, 1))
    scalers["solar"].fit(df_train[config.SOLAR_FEATURES])
    print(f"  ✓ Solar scaler (MinMax [0,1])  : {config.SOLAR_FEATURES}")

    # Meteorological  → z-score
    scalers["met"] = StandardScaler()
    scalers["met"].fit(df_train[config.MET_FEATURES])
    print(f"  ✓ Met scaler   (Standard)      : {config.MET_FEATURES}")

    # Precipitation / snow  → [0, 1]
    scalers["precip"] = MinMaxScaler(feature_range=(0, 1))
    scalers["precip"].fit(df_train[config.PRECIP_FEATURES])
    print(f"  ✓ Precip scaler (MinMax [0,1]) : {config.PRECIP_FEATURES}")

    # Angle features  → [-1, 1]   (sunheight ranges −90°→+90°, winddir 0°→360°)
    scalers["angle"] = MinMaxScaler(feature_range=(-1, 1))
    scalers["angle"].fit(df_train[config.ANGLE_FEATURES])
    print(f"  ✓ Angle scaler (MinMax [-1,1]) : {config.ANGLE_FEATURES}")

    # Target  → [0, 1]
    scalers["target"] = MinMaxScaler(feature_range=(0, 1))
    scalers["target"].fit(df_train[[config.TARGET]])
    print(f"  ✓ Target scaler (MinMax [0,1]) : {config.TARGET}")

    print(f"  ✓ Cyclic / binary features: no scaling required")

    return scalers


def apply_scaling(df: pd.DataFrame, scalers: dict, config: Config) -> pd.DataFrame:
    """Apply pre-fitted scalers to a split."""
    df_sc = df.copy()
    df_sc[config.SOLAR_FEATURES]  = scalers["solar"].transform(df[config.SOLAR_FEATURES])
    df_sc[config.MET_FEATURES]    = scalers["met"].transform(df[config.MET_FEATURES])
    df_sc[config.PRECIP_FEATURES] = scalers["precip"].transform(df[config.PRECIP_FEATURES])
    df_sc[config.ANGLE_FEATURES]  = scalers["angle"].transform(df[config.ANGLE_FEATURES])
    df_sc[[config.TARGET]]        = scalers["target"].transform(df[[config.TARGET]])
    return df_sc


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — SEQUENCE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(data: np.ndarray, target: np.ndarray,
                     lookback: int = 24, horizon: int = 1):
    """
    Build (X, y) pairs for time-series modelling.

    X[i] = rows [i-lookback … i-1]  →  y[i] = target at step i + horizon - 1
    """
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback : i])
        y.append(target[i + horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def generate_all_sequences(df_train, df_val, df_test,
                            config: Config, feature_cols: list):
    """Generate sequences for all three splits."""
    print("\n" + "=" * 70)
    print("STAGE 6: SEQUENCE GENERATION")
    print("=" * 70)
    print(f"  Lookback : {config.LOOKBACK_WINDOW} h | Horizon: {config.FORECAST_HORIZON} h")
    print(f"  Features : {len(feature_cols)}")

    results = {}
    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X, y = create_sequences(
            split[feature_cols].values,
            split[config.TARGET].values,
            config.LOOKBACK_WINDOW,
            config.FORECAST_HORIZON,
        )
        results[f"X_{name}"] = X
        results[f"y_{name}"] = y
        print(f"  {name:5s}  X: {X.shape}  y: {y.shape}")

    # Assertions
    for split in ["train", "val", "test"]:
        assert results[f"X_{split}"].shape[1] == config.LOOKBACK_WINDOW
        assert results[f"X_{split}"].shape[2] == len(feature_cols)
    print("  ✓ Shape assertions passed")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — QUALITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_sequences(seqs: dict) -> bool:
    """Run NaN/Inf/range/size checks on all sequence arrays."""
    print("\n" + "=" * 70)
    print("STAGE 7: QUALITY VALIDATION")
    print("=" * 70)

    passed = 0
    total  = 5

    # 1. No NaN
    no_nan = all(np.isnan(v).sum() == 0 for v in seqs.values())
    print(f"  {'✓' if no_nan else '✗'} No NaN values")
    passed += no_nan

    # 2. No Inf
    no_inf = all(np.isinf(v).sum() == 0 for v in seqs.values())
    print(f"  {'✓' if no_inf else '✗'} No Inf values")
    passed += no_inf

    # 3. Target in [0, 1.5]  (small slack for potential scaler overshoot)
    tgt_ok = all(
        seqs[f"y_{s}"].min() >= 0 and seqs[f"y_{s}"].max() <= 1.5
        for s in ["train", "val", "test"]
    )
    print(f"  {'✓' if tgt_ok else '✗'} Target values in expected range [0, ~1]")
    passed += tgt_ok

    # 4. Consistent shapes
    shape_ok = (
        seqs["X_train"].shape[1:] == seqs["X_val"].shape[1:] ==
        seqs["X_test"].shape[1:]
    )
    print(f"  {'✓' if shape_ok else '✗'} Consistent sequence shapes across splits")
    passed += shape_ok

    # 5. Sufficient training samples
    enough = len(seqs["X_train"]) >= 500
    print(f"  {'✓' if enough else '✗'} ≥500 training sequences ({len(seqs['X_train']):,})")
    passed += enough

    print(f"\n  {passed}/{total} checks passed {'✓' if passed == total else '⚠'}")
    return passed == total


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — SAVE PREPROCESSED DATA
# ══════════════════════════════════════════════════════════════════════════════

def save_preprocessed(seqs: dict, scalers: dict,
                       feature_cols: list, config: Config):
    """Save arrays, scalers, and metadata to disk."""
    print("\n" + "=" * 70)
    print("STAGE 8: SAVING PREPROCESSED DATA")
    print("=" * 70)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Arrays
    for key, arr in seqs.items():
        np.save(config.OUTPUT_DIR / f"{key}.npy", arr)
    print(f"  ✓ Saved {len(seqs)} numpy arrays → {config.OUTPUT_DIR}")

    # Scalers
    with open(config.OUTPUT_DIR / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    print(f"  ✓ Saved scalers (pickle)")

    # Metadata
    meta = {
        "dataset"          : "Radekhiv, Ukraine",
        "target"           : config.TARGET,
        "feature_columns"  : feature_cols,
        "n_features"       : len(feature_cols),
        "lookback_window"  : config.LOOKBACK_WINDOW,
        "forecast_horizon" : config.FORECAST_HORIZON,
        "train_size"       : int(len(seqs["X_train"])),
        "val_size"         : int(len(seqs["X_val"])),
        "test_size"        : int(len(seqs["X_test"])),
        "preprocessing_ts" : pd.Timestamp.now().isoformat(),
        "feature_groups"   : {
            "solar"   : config.SOLAR_FEATURES,
            "met"     : config.MET_FEATURES,
            "precip"  : config.PRECIP_FEATURES,
            "angle"   : config.ANGLE_FEATURES,
            "cyclic"  : ["hour_sin","hour_cos","month_sin","month_cos",
                         "doy_sin","doy_cos"],
            "flags"   : ["is_daytime","is_peak_hours"],
            "interact": ["cloud_rad","temp_rad"],
        },
    }
    with open(config.OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Saved metadata.json")

    # File listing
    print(f"\n  Files in {config.OUTPUT_DIR}:")
    for fp in sorted(config.OUTPUT_DIR.iterdir()):
        print(f"    {fp.name:25s}  {fp.stat().st_size / 1024:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — DUAL-STREAM CNN-LSTM MODEL  (DSCLANet)
# ══════════════════════════════════════════════════════════════════════════════

class DSCLANet:
    """
    Dual-Stream CNN-LSTM with Attention Network
    ─────────────────────────────────────────────
    Stream 1  : 3× Conv1D  → spatial / local-pattern features
    Stream 2  : 2× LSTM    → temporal / sequential features
    Fusion    : Concatenate both streams
    Attention : Self-attention (element-wise soft-weighting)
    Head      : 3× Dense → 1 output (linear regression)

    Reference : "Solar Power Prediction Using Dual Stream CNN-LSTM Architecture"
                PMC9864442
    """

    def __init__(self, input_shape,
                 cnn_filters=[32, 64, 128],
                 cnn_kernels=[5, 3, 1],
                 lstm_units=[100, 100],
                 dense_units=[64, 32, 12],
                 dropout_rate=0.2,
                 learning_rate=0.001):
        self.input_shape   = input_shape
        self.cnn_filters   = cnn_filters
        self.cnn_kernels   = cnn_kernels
        self.lstm_units    = lstm_units
        self.dense_units   = dense_units
        self.dropout_rate  = dropout_rate
        self.learning_rate = learning_rate
        self.model         = None
        self.history       = None

    # ── Sub-network builders ─────────────────────────────────────────────

    def _cnn_stream(self, x):
        """3× Conv1D with dropout — extracts local temporal patterns."""
        for i, (f, k) in enumerate(zip(self.cnn_filters, self.cnn_kernels)):
            x = layers.Conv1D(f, k, padding="same", activation="relu",
                              name=f"cnn_conv{i+1}")(x)
            if i < len(self.cnn_filters) - 1:         # dropout after first two only
                x = layers.Dropout(self.dropout_rate,
                                   name=f"cnn_drop{i+1}")(x)
        return layers.Flatten(name="cnn_flat")(x)

    def _lstm_stream(self, x):
        """2× LSTM — captures long-range temporal dependencies."""
        x = layers.LSTM(self.lstm_units[0], return_sequences=True,
                        name="lstm1")(x)
        x = layers.Dropout(self.dropout_rate, name="lstm_drop1")(x)
        x = layers.LSTM(self.lstm_units[1], return_sequences=False,
                        name="lstm2")(x)
        return layers.Dropout(self.dropout_rate, name="lstm_drop2")(x)

    def _attention(self, x):
        """Element-wise self-attention: learns which fused features matter most."""
        scores  = layers.Dense(x.shape[-1], activation="tanh",
                               name="attn_scores")(x)
        weights = layers.Dense(x.shape[-1], activation="softmax",
                               name="attn_weights")(scores)
        return layers.Multiply(name="attn_apply")([x, weights])

    # ── Model builder ────────────────────────────────────────────────────

    def build(self):
        inputs = layers.Input(shape=self.input_shape, name="input")

        cnn_out  = self._cnn_stream(inputs)
        lstm_out = self._lstm_stream(inputs)
        fused    = layers.Concatenate(name="fusion")([cnn_out, lstm_out])
        attended = self._attention(fused)

        x = attended
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation="relu", name=f"dense{i+1}")(x)
            x = layers.Dropout(self.dropout_rate, name=f"dense_drop{i+1}")(x)

        output = layers.Dense(1, activation="linear", name="output")(x)

        self.model = Model(inputs, output, name="DSCLANet_Radekhiv")
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss="mse",
            metrics=["mae",
                     tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        print("✓ DSCLANet built and compiled")
        self.model.summary()
        return self.model

    # ── Training ─────────────────────────────────────────────────────────

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32, patience=15,
            model_dir=Path("radekhiv_models")):
        if self.model is None:
            raise RuntimeError("Call build() first.")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(filepath=str(model_dir / "best_model.keras"),
                            monitor="val_loss", save_best_only=True, verbose=1),
        ]

        print(f"\n  Training: {len(X_train):,} samples  |  "
              f"Val: {len(X_val):,}  |  Batch: {batch_size}  |  "
              f"Max epochs: {epochs}\n")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        print("\n✓ Training complete")
        return self.history

    # ── Prediction & evaluation ──────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    def evaluate_in_watts(self, X_test, y_test_scaled,
                          target_scaler, label="TEST"):
        """
        Inverse-transform predictions to original generation units and
        compute MAE, RMSE, R² in those units.
        """
        y_pred_scaled = self.predict(X_test)

        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(
            y_test_scaled.reshape(-1, 1)).flatten()

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        print(f"\n  ── {label} METRICS (original generation units) ──")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R²   : {r2:.6f}")

        return {"mae": mae, "rmse": rmse, "r2": r2,
                "y_true": y_true, "y_pred": y_pred}

    def save(self, path):
        self.model.save(str(path))
        print(f"✓ Model saved → {path}")

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.model = keras.models.load_model(str(path))
        obj.history = None
        print(f"✓ Model loaded from {path}")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("   RADEKHIV — DUAL-STREAM CNN-LSTM (DSCLANet) PIPELINE")
    print("=" * 70)

    cfg = Config()
    np.random.seed(cfg.RANDOM_SEED)
    tf.random.set_seed(cfg.RANDOM_SEED)

    # ── Stages 1-3 ──────────────────────────────────────────────────────
    df = load_and_validate(cfg)
    df = feature_engineering(df)
    df = filter_data(df, cfg)

    # ── Stage 4 ─────────────────────────────────────────────────────────
    df_train, df_val, df_test = temporal_split(df, cfg)

    # ── Stage 5 ─────────────────────────────────────────────────────────
    scalers = create_and_fit_scalers(df_train, cfg)
    df_train_sc = apply_scaling(df_train, scalers, cfg)
    df_val_sc   = apply_scaling(df_val,   scalers, cfg)
    df_test_sc  = apply_scaling(df_test,  scalers, cfg)

    # Define full ordered feature list for model input
    feature_columns = (
        cfg.SOLAR_FEATURES   +   # 3  solar / irradiance
        cfg.MET_FEATURES     +   # 9  meteorological
        cfg.PRECIP_FEATURES  +   # 4  precipitation / snow
        cfg.ANGLE_FEATURES   +   # 2  angular
        ["hour_sin", "hour_cos",
         "month_sin", "month_cos",
         "doy_sin", "doy_cos"]   # 6  cyclic time
        + ["is_daytime", "is_peak_hours"]   # 2  binary flags
        + ["cloud_rad", "temp_rad"]         # 2  interactions
    )   # Total: 28 features

    print(f"\n  Feature columns ({len(feature_columns)} total):")
    for i, col in enumerate(feature_columns, 1):
        print(f"    {i:2d}. {col}")

    # ── Stages 6-7 ──────────────────────────────────────────────────────
    seqs = generate_all_sequences(
        df_train_sc, df_val_sc, df_test_sc, cfg, feature_columns)
    all_valid = validate_sequences(seqs)

    # ── Stage 8 ─────────────────────────────────────────────────────────
    save_preprocessed(seqs, scalers, feature_columns, cfg)

    if not all_valid:
        print("\n⚠  Validation failed — aborting training. Review errors above.")
        return

    # ── Stage 9 — Build & Train ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 9: BUILD & TRAIN DSCLANet")
    print("=" * 70)

    input_shape = (cfg.LOOKBACK_WINDOW, len(feature_columns))
    net = DSCLANet(
        input_shape   = input_shape,
        cnn_filters   = cfg.CNN_FILTERS,
        cnn_kernels   = cfg.CNN_KERNELS,
        lstm_units    = cfg.LSTM_UNITS,
        dense_units   = cfg.DENSE_UNITS,
        dropout_rate  = cfg.DROPOUT_RATE,
        learning_rate = cfg.LEARNING_RATE,
    )
    net.build()
    net.fit(
        seqs["X_train"], seqs["y_train"],
        seqs["X_val"],   seqs["y_val"],
        epochs     = cfg.MAX_EPOCHS,
        batch_size = cfg.BATCH_SIZE,
        patience   = cfg.PATIENCE,
        model_dir  = cfg.MODEL_DIR,
    )

    # ── Stage 10 — Evaluate ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 10: EVALUATION")
    print("=" * 70)

    val_metrics  = net.evaluate_in_watts(
        seqs["X_val"],  seqs["y_val"],  scalers["target"], label="VALIDATION")
    test_metrics = net.evaluate_in_watts(
        seqs["X_test"], seqs["y_test"], scalers["target"], label="TEST")

    # Save final metrics
    all_metrics = {"validation": val_metrics, "test": test_metrics}
    # Remove numpy arrays from the serialisable dict
    for split in all_metrics.values():
        split.pop("y_true", None)
        split.pop("y_pred", None)

    metrics_path = cfg.MODEL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved → {metrics_path}")

    # Save final model
    net.save(cfg.MODEL_DIR / "final_model.keras")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Preprocessed data  : {cfg.OUTPUT_DIR}/")
    print(f"  Best model weights : {cfg.MODEL_DIR}/best_model.keras")
    print(f"  Final model        : {cfg.MODEL_DIR}/final_model.keras")
    print(f"  Metrics            : {cfg.MODEL_DIR}/metrics.json")
    print("\nTo load scalers for inference:")
    print(f"  with open('{cfg.OUTPUT_DIR}/scalers.pkl','rb') as f:")
    print(f"      scalers = pickle.load(f)")
    print("\nTo load the trained model:")
    print(f"  net = DSCLANet.load('{cfg.MODEL_DIR}/best_model.keras')")


if __name__ == "__main__":
    main()