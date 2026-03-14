"""
visualize_predictions.py
Plots 1-hour-ahead predicted vs actual PV power output
Run this after dual_stream.py has completed training.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
from tensorflow import keras

DATA_DIR = Path("/Users/amruthaj/Documents/GitHub/Solar-Prediction-with-ML-DL-Models/PVGIS+NASA_with_DL/preprocessing/preprocessed_data")

# ── Load data & model ──────────────────────────────────────────────────────
X_test = np.load(DATA_DIR / 'X_test.npy')
y_test = np.load(DATA_DIR / 'y_test.npy')

with open(DATA_DIR / 'scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

model = keras.models.load_model(DATA_DIR / 'final_model.keras')

# ── Predict & inverse transform ────────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = scalers['target'].inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_actual = scalers['target'].inverse_transform(y_test.reshape(-1,1)).flatten()
errors = y_actual - y_pred

# ── Plot 1: Full test period overview ─────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Actual vs Predicted
axes[0].plot(y_actual, label='Actual', color='steelblue', linewidth=0.8, alpha=0.9)
axes[0].plot(y_pred,   label='Predicted (t+1h)', color='orange', linewidth=0.8, alpha=0.8)
axes[0].set_title('1-Hour-Ahead PV Power Prediction — Full Test Period', fontsize=13)
axes[0].set_ylabel('Power (W)')
axes[0].legend()
axes[0].set_xlim(0, len(y_actual))

# Residuals
axes[1].plot(errors, color='tomato', linewidth=0.6, alpha=0.8)
axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
axes[1].fill_between(range(len(errors)), errors, 0, alpha=0.3, color='tomato')
axes[1].set_title('Residuals (Actual − Predicted)', fontsize=13)
axes[1].set_ylabel('Error (W)')
axes[1].set_xlabel('Test Sample Index')

# Error distribution
axes[2].hist(errors, bins=60, color='mediumpurple', edgecolor='white', linewidth=0.4)
axes[2].axvline(errors.mean(), color='red', linestyle='--', label=f'Mean = {errors.mean():.1f} W')
axes[2].axvline(errors.mean() + errors.std(), color='orange', linestyle=':', label=f'±1σ = {errors.std():.1f} W')
axes[2].axvline(errors.mean() - errors.std(), color='orange', linestyle=':')
axes[2].set_title('Error Distribution', fontsize=13)
axes[2].set_xlabel('Error (W)')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.savefig(DATA_DIR / 'prediction_overview.png', dpi=150)
plt.close()
print("✓ Saved: prediction_overview.png")

# ── Plot 2: 7-day zoomed window ────────────────────────────────────────────
# Shows the t+1h predictions in detail (168 daytime hours ≈ ~2–3 weeks of daytime)
window = 168
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_actual[:window], label='Actual',           color='steelblue', linewidth=1.5)
ax.plot(y_pred[:window],   label='Predicted (t+1h)', color='orange',    linewidth=1.5, linestyle='--')
ax.fill_between(range(window), y_actual[:window], y_pred[:window],
                alpha=0.2, color='red', label='Error region')
ax.set_title('1-Hour-Ahead Predictions — First 168 Test Samples (Zoomed)', fontsize=13)
ax.set_xlabel('Sample Index (daytime hours)')
ax.set_ylabel('PV Power (W)')
ax.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / 'prediction_zoomed.png', dpi=150)
plt.close()
print("✓ Saved: prediction_zoomed.png")

# ── Plot 3: Scatter — Actual vs Predicted ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_actual, y_pred, alpha=0.3, s=8, color='steelblue', edgecolors='none')
lims = [0, max(y_actual.max(), y_pred.max()) * 1.05]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual Power (W)')
ax.set_ylabel('Predicted Power (W)')
ax.set_title('Actual vs Predicted — Test Set', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(DATA_DIR / 'scatter_actual_vs_pred.png', dpi=150)
plt.close()
print("✓ Saved: scatter_actual_vs_pred.png")

# ── Print summary table ────────────────────────────────────────────────────
mae  = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
ss_res = np.sum(errors**2)
ss_tot = np.sum((y_actual - y_actual.mean())**2)
r2 = 1 - ss_res / ss_tot

print("\n1-Hour-Ahead Forecast Summary")
print("="*40)
print(f"  MAE   : {mae:.2f} W")
print(f"  RMSE  : {rmse:.2f} W")
print(f"  R²    : {r2:.4f}")
print(f"  Max overestimate : {errors.min():.1f} W")
print(f"  Max underestimate: {errors.max():.1f} W")
print("="*40)