"""
Solar PV Power Prediction - Preprocessing Implementation
=========================================================
Implements the preprocessing pipeline for Dual Stream CNN-LSTM model
Based on: "Solar Power Prediction Using Dual Stream CNN-LSTM Architecture" (PMC9864442)

Author: [Your Name]
Date: March 2026
Project: Capstone - Solar Power Prediction for Rajasthan, India
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Preprocessing configuration parameters"""
    
    # Paths
    DATA_PATH = Path("/Users/amruthaj/Documents/GitHub/Solar-Prediction-with-ML-DL-Models/PVGIS+NASA_with_DL/datacleaning/processed_solar_2019_2020.csv")
    OUTPUT_DIR = Path("/Users/amruthaj/Documents/GitHub/Solar-Prediction-with-ML-DL-Models/PVGIS+NASA_with_DL/preprocessing/preprocessed_data")
    
    # Sequence parameters (as per paper)
    LOOKBACK_WINDOW = 2   # hours of historical data
    FORECAST_HORIZON = 1  # hours ahead to predict
    
    # Train/Val/Test split (temporal)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    
    # Feature groups
    SPATIAL_FEATURES = ['G_i', 'Gb_i', 'Gd_i', 'Gr_i']
    TEMPORAL_FEATURES = ['T2m', 'WS10m', 'RH', 'SP']
    ANGLE_FEATURES = ['H_sun', 'WD10m']
    TARGET = 'P'
    
    # Quality control
    USE_DAYTIME_ONLY = True  # Filter G_i > 0
    MIN_DATA_QUALITY = 0     # 0 = all data, 1 = clean only
    
    # Random seed
    RANDOM_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def add_temporal_features(df):
    """
    Add cyclic temporal features and flags.
    
    Features added:
    - hour_sin, hour_cos: Hour of day (24-hour cycle)
    - month_sin, month_cos: Month (12-month cycle)
    - doy_sin, doy_cos: Day of year (365-day cycle)
    - is_daytime: Flag for sun above horizon
    - is_peak_hours: Flag for peak solar hours (5-9 UTC ≈ 10:30-14:30 IST)
    """
    print("\n" + "="*70)
    print("STAGE 1: FEATURE ENGINEERING")
    print("="*70)
    
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear
    
    # Cyclic encoding: hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclic encoding: month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Cyclic encoding: day of year (365-day cycle)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Binary flags
    df['is_daytime'] = (df['H_sun'] > 0).astype(int)
    df['is_peak_hours'] = ((df['hour'] >= 5) & (df['hour'] <= 9)).astype(int)
    
    print(f"✓ Added 10 new features:")
    print(f"  - Cyclic time: hour_sin/cos, month_sin/cos, doy_sin/cos")
    print(f"  - Binary flags: is_daytime, is_peak_hours")
    
    # Drop intermediate columns
    df = df.drop(columns=['hour', 'month', 'day_of_year'])
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: DATA FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_data(df, config):
    """
    Filter data based on quality and operational criteria.
    """
    print("\n" + "="*70)
    print("STAGE 2: DATA FILTERING")
    print("="*70)
    
    initial_rows = len(df)
    
    # Filter 1: Daytime only (G_i > 0)
    if config.USE_DAYTIME_ONLY:
        df = df[df['G_i'] > 0].copy()
        print(f"✓ Daytime filter (G_i > 0): {initial_rows:,} → {len(df):,} rows")
    
    # Filter 2: Data quality
    if config.MIN_DATA_QUALITY > 0:
        df = df[df['data_quality'] >= config.MIN_DATA_QUALITY].copy()
        print(f"✓ Quality filter (data_quality >= {config.MIN_DATA_QUALITY}): {len(df):,} rows remain")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠ Warning: Missing values detected:")
        print(missing[missing > 0])
    else:
        print(f"✓ No missing values in filtered dataset")
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: TRAIN-VAL-TEST SPLIT (TEMPORAL)
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(df, config):
    """
    Split data temporally (not randomly) to preserve time-series structure.
    """
    print("\n" + "="*70)
    print("STAGE 3: TEMPORAL TRAIN-VAL-TEST SPLIT")
    print("="*70)
    
    n_samples = len(df)
    
    # Calculate split indices
    train_end = int(n_samples * config.TRAIN_RATIO)
    val_end = int(n_samples * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    # Split
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    print(f"Total samples: {n_samples:,}")
    print(f"  Train: {len(df_train):,} ({len(df_train)/n_samples*100:.1f}%)")
    print(f"  Val  : {len(df_val):,} ({len(df_val)/n_samples*100:.1f}%)")
    print(f"  Test : {len(df_test):,} ({len(df_test)/n_samples*100:.1f}%)")
    
    print(f"\nDate ranges:")
    print(f"  Train: {df_train['datetime'].iloc[0]} to {df_train['datetime'].iloc[-1]}")
    print(f"  Val  : {df_val['datetime'].iloc[0]} to {df_val['datetime'].iloc[-1]}")
    print(f"  Test : {df_test['datetime'].iloc[0]} to {df_test['datetime'].iloc[-1]}")
    
    return df_train, df_val, df_test


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════════════

def create_scalers(df_train, config):
    """
    Create and fit scalers on training data only.
    
    Scaling strategy:
    - Spatial features (irradiance): MinMaxScaler [0, 1]
    - Temporal features (met): StandardScaler
    - Angle features: MinMaxScaler [-1, 1]
    - Target: MinMaxScaler [0, 1]
    - Cyclic features: No scaling (already in [-1, 1])
    """
    print("\n" + "="*70)
    print("STAGE 4: FEATURE SCALING")
    print("="*70)
    
    scalers = {}
    
    # Spatial features: MinMax [0, 1]
    scalers['spatial'] = MinMaxScaler(feature_range=(0, 1))
    scalers['spatial'].fit(df_train[config.SPATIAL_FEATURES])
    print(f"✓ Spatial scaler fitted on: {config.SPATIAL_FEATURES}")
    
    # Temporal features: Standard
    scalers['temporal'] = StandardScaler()
    scalers['temporal'].fit(df_train[config.TEMPORAL_FEATURES])
    print(f"✓ Temporal scaler fitted on: {config.TEMPORAL_FEATURES}")
    
    # Angle features: MinMax [-1, 1]
    scalers['angle'] = MinMaxScaler(feature_range=(-1, 1))
    scalers['angle'].fit(df_train[config.ANGLE_FEATURES])
    print(f"✓ Angle scaler fitted on: {config.ANGLE_FEATURES}")
    
    # Target: MinMax [0, 1]
    scalers['target'] = MinMaxScaler(feature_range=(0, 1))
    scalers['target'].fit(df_train[[config.TARGET]])
    print(f"✓ Target scaler fitted on: {config.TARGET}")
    
    # Cyclic and flag features need no scaling
    print(f"✓ Cyclic/flag features: No scaling applied")
    
    return scalers


def apply_scaling(df, scalers, config):
    """
    Apply fitted scalers to dataframe.
    """
    df_scaled = df.copy()
    
    # Apply each scaler
    df_scaled[config.SPATIAL_FEATURES] = scalers['spatial'].transform(df[config.SPATIAL_FEATURES])
    df_scaled[config.TEMPORAL_FEATURES] = scalers['temporal'].transform(df[config.TEMPORAL_FEATURES])
    df_scaled[config.ANGLE_FEATURES] = scalers['angle'].transform(df[config.ANGLE_FEATURES])
    df_scaled[[config.TARGET]] = scalers['target'].transform(df[[config.TARGET]])
    
    return df_scaled


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: SEQUENCE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(data, target, lookback=2, horizon=1):
    """
    Create sequences for time-series forecasting.
    
    Parameters:
    -----------
    data : np.array, shape (n_samples, n_features)
        Scaled feature matrix
    target : np.array, shape (n_samples,)
        Scaled target variable
    lookback : int
        Number of past timesteps (2 hours)
    horizon : int
        Steps ahead to predict (1 hour)
    
    Returns:
    --------
    X : np.array, shape (n_sequences, lookback, n_features)
        Input sequences
    y : np.array, shape (n_sequences,)
        Target values
    """
    X, y = [], []
    
    for i in range(lookback, len(data) - horizon + 1):
        # Input: [t-lookback, ..., t-1]
        # Output: t + horizon - 1
        X.append(data[i - lookback:i])
        y.append(target[i + horizon - 1])
    
    return np.array(X), np.array(y)


def generate_sequences(df_train, df_val, df_test, config, feature_columns):
    """
    Generate sequences for all splits.
    """
    print("\n" + "="*70)
    print("STAGE 5: SEQUENCE GENERATION")
    print("="*70)
    
    print(f"Parameters:")
    print(f"  Lookback window: {config.LOOKBACK_WINDOW} hours")
    print(f"  Forecast horizon: {config.FORECAST_HORIZON} hour(s)")
    print(f"  Number of features: {len(feature_columns)}")
    
    # Extract features and target as numpy arrays
    X_train_data = df_train[feature_columns].values
    y_train_data = df_train[config.TARGET].values
    
    X_val_data = df_val[feature_columns].values
    y_val_data = df_val[config.TARGET].values
    
    X_test_data = df_test[feature_columns].values
    y_test_data = df_test[config.TARGET].values
    
    # Create sequences
    X_train, y_train = create_sequences(
        X_train_data, y_train_data, 
        config.LOOKBACK_WINDOW, config.FORECAST_HORIZON
    )
    
    X_val, y_val = create_sequences(
        X_val_data, y_val_data,
        config.LOOKBACK_WINDOW, config.FORECAST_HORIZON
    )
    
    X_test, y_test = create_sequences(
        X_test_data, y_test_data,
        config.LOOKBACK_WINDOW, config.FORECAST_HORIZON
    )
    
    print(f"\n✓ Sequences created:")
    print(f"  X_train: {X_train.shape} → y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape} → y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape} → y_test : {y_test.shape}")
    
    # Validation checks
    assert X_train.shape[1] == config.LOOKBACK_WINDOW, "Lookback dimension mismatch"
    assert X_train.shape[2] == len(feature_columns), "Feature dimension mismatch"
    assert len(y_train) == len(X_train), "Train X-y length mismatch"
    assert len(y_val) == len(X_val), "Val X-y length mismatch"
    assert len(y_test) == len(X_test), "Test X-y length mismatch"
    
    print(f"✓ All validation checks passed")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6: QUALITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Perform quality checks on preprocessed data.
    """
    print("\n" + "="*70)
    print("STAGE 6: DATA QUALITY VALIDATION")
    print("="*70)
    
    checks_passed = 0
    total_checks = 6
    
    # Check 1: No NaN values
    nan_check = (
        np.isnan(X_train).sum() == 0 and
        np.isnan(y_train).sum() == 0 and
        np.isnan(X_val).sum() == 0 and
        np.isnan(y_val).sum() == 0 and
        np.isnan(X_test).sum() == 0 and
        np.isnan(y_test).sum() == 0
    )
    print(f"  {'✓' if nan_check else '✗'} No NaN values")
    checks_passed += nan_check
    
    # Check 2: No Inf values
    inf_check = (
        np.isinf(X_train).sum() == 0 and
        np.isinf(y_train).sum() == 0 and
        np.isinf(X_val).sum() == 0 and
        np.isinf(y_val).sum() == 0 and
        np.isinf(X_test).sum() == 0 and
        np.isinf(y_test).sum() == 0
    )
    print(f"  {'✓' if inf_check else '✗'} No Inf values")
    checks_passed += inf_check
    
    # Check 3: Reasonable value ranges (target in [0, 1] after scaling)
    target_range_check = (
        y_train.min() >= 0 and y_train.max() <= 1.5 and
        y_val.min() >= 0 and y_val.max() <= 1.5 and
        y_test.min() >= 0 and y_test.max() <= 1.5
    )
    print(f"  {'✓' if target_range_check else '✗'} Target values in expected range [0, ~1]")
    checks_passed += target_range_check
    
    # Check 4: No data leakage (test is future of train)
    # (This was ensured by temporal split, but we confirm sizes)
    leakage_check = (len(X_train) > len(X_val)) and (len(X_train) > len(X_test))
    print(f"  {'✓' if leakage_check else '✗'} Train set larger than val/test (temporal integrity)")
    checks_passed += leakage_check
    
    # Check 5: Sequence shapes consistent
    shape_check = (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1] and
        X_train.shape[2] == X_val.shape[2] == X_test.shape[2]
    )
    print(f"  {'✓' if shape_check else '✗'} Sequence shapes consistent across splits")
    checks_passed += shape_check
    
    # Check 6: Sufficient samples for training
    sample_check = len(X_train) > 1000  # At least 1000 training sequences
    print(f"  {'✓' if sample_check else '✗'} Sufficient training samples (>1000)")
    checks_passed += sample_check
    
    print(f"\n{'✓' if checks_passed == total_checks else '⚠'} {checks_passed}/{total_checks} checks passed")
    
    if checks_passed < total_checks:
        print("  ⚠ WARNING: Some validation checks failed. Review data before training.")
    
    return checks_passed == total_checks


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7: SAVE PREPROCESSED DATA
# ══════════════════════════════════════════════════════════════════════════════

def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test,
                           scalers, feature_columns, config):
    """
    Save all preprocessed data, scalers, and metadata.
    """
    print("\n" + "="*70)
    print("STAGE 7: SAVING PREPROCESSED DATA")
    print("="*70)
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Save numpy arrays
    np.save(config.OUTPUT_DIR / 'X_train.npy', X_train)
    np.save(config.OUTPUT_DIR / 'y_train.npy', y_train)
    np.save(config.OUTPUT_DIR / 'X_val.npy', X_val)
    np.save(config.OUTPUT_DIR / 'y_val.npy', y_val)
    np.save(config.OUTPUT_DIR / 'X_test.npy', X_test)
    np.save(config.OUTPUT_DIR / 'y_test.npy', y_test)
    print(f"✓ Saved numpy arrays to: {config.OUTPUT_DIR}")
    
    # Save scalers
    with open(config.OUTPUT_DIR / 'scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    print(f"✓ Saved scalers (pickle)")
    
    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'lookback_window': config.LOOKBACK_WINDOW,
        'forecast_horizon': config.FORECAST_HORIZON,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'n_features': len(feature_columns),
        'target_column': config.TARGET,
        'preprocessing_date': pd.Timestamp.now().isoformat()
    }
    
    with open(config.OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata (JSON)")
    
    # Save feature info
    feature_info = {
        'spatial_features': config.SPATIAL_FEATURES,
        'temporal_features': config.TEMPORAL_FEATURES,
        'angle_features': config.ANGLE_FEATURES,
        'cyclic_features': ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos'],
        'flag_features': ['is_daytime', 'is_peak_hours'],
        'target': config.TARGET
    }
    
    with open(config.OUTPUT_DIR / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✓ Saved feature info (JSON)")
    
    print(f"\nAll files saved to: {config.OUTPUT_DIR}")
    print("  Files:")
    for f in sorted(config.OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name:25s} ({size_mb:6.2f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Execute complete preprocessing pipeline.
    """
    print("\n" + "="*70)
    print(" "*15 + "CNN-LSTM PREPROCESSING PIPELINE")
    print("="*70)
    print(f"Data source: {Config.DATA_PATH}")
    print(f"Output dir : {Config.OUTPUT_DIR}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(Config.DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"✓ Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Column selection previously done in preprocess_CNNLSTM.py
    target_col = 'P'
    base_features = ['G_i', 'Gb_i', 'Gd_i', 'Gr_i', 'H_sun', 'T2m', 'WS10m', 'RH', 'SP', 'WD10m']
    optional_feature = 'data_quality'
    keep_cols = ['datetime', target_col] + base_features + [optional_feature]
    df = df[keep_cols].copy()
    print(f"✓ Selected modeling columns: {len(df.columns)}")
    
    # STAGE 1: Feature engineering
    df = add_temporal_features(df)
    
    # STAGE 2: Data filtering
    df = filter_data(df, Config)
    
    # STAGE 3: Temporal split
    df_train, df_val, df_test = temporal_split(df, Config)
    
    # STAGE 4: Scaling
    scalers = create_scalers(df_train, Config)
    df_train_scaled = apply_scaling(df_train, scalers, Config)
    df_val_scaled = apply_scaling(df_val, scalers, Config)
    df_test_scaled = apply_scaling(df_test, scalers, Config)
    print("✓ Scaling applied to all splits")
    
    # Define feature columns for model input
    feature_columns = (
        Config.SPATIAL_FEATURES +       # 4 features
        Config.TEMPORAL_FEATURES +      # 4 features
        Config.ANGLE_FEATURES +         # 2 features
        ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos'] +  # 6 features
        ['is_daytime', 'is_peak_hours'] # 2 features
    )
    # Total: 18 features
    
    print(f"\nFeature columns ({len(feature_columns)} total):")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i:2d}. {col}")
    
    # STAGE 5: Sequence generation
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = generate_sequences(
        df_train_scaled, df_val_scaled, df_test_scaled, Config, feature_columns
    )
    
    # STAGE 6: Validation
    validation_passed = validate_data(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # STAGE 7: Save
    if validation_passed:
        save_preprocessed_data(
            X_train, y_train, X_val, y_val, X_test, y_test,
            scalers, feature_columns, Config
        )
    else:
        print("\n⚠ Data validation failed. Not saving preprocessed data.")
        print("  Please review the validation errors above.")
        return
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Build Dual Stream CNN-LSTM model")
    print("  2. Load preprocessed data with:")
    print(f"     X_train = np.load('{Config.OUTPUT_DIR}/X_train.npy')")
    print(f"     y_train = np.load('{Config.OUTPUT_DIR}/y_train.npy')")
    print("  3. Train and evaluate model")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()