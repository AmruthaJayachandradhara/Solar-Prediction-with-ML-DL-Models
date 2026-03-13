"""
Exploratory Data Analysis (EDA) — Solar PV Power Prediction
=============================================================
Dataset : processed_solar_2019_2020.csv
Location: Rajasthan, India  (lat=26.81, lon=73.77)
Target  : P (PV power output in Watts, 1 kWp system)

Visualization Tools : Matplotlib, Seaborn, SHAP
Structure follows standard EDA workflow:
  Step 1 — Load & Inspect
  Step 2 — Univariate Analysis
  Step 3 — Bivariate Analysis
  Step 4 — Group / Temporal Analysis
  Step 5 — Multivariate Analysis (Correlation Heatmap)
  Step 6 — Outlier & Skewness Analysis
  Step 7 — SHAP Feature Importance & Dependence

Outputs: 15+ PNG plots saved to Capstone/eda_plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import warnings
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_CSV, EDA_PLOTS_DIR, ensure_dirs

warnings.filterwarnings("ignore")

ensure_dirs()
OUT = EDA_PLOTS_DIR

# Set global style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD & INSPECT THE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: LOAD & INSPECT THE DATA")
print("=" * 65)

df = pd.read_csv(PROCESSED_CSV)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

# Add time features
df["hour"]   = df["datetime"].dt.hour
df["month"]  = df["datetime"].dt.month
df["year"]   = df["datetime"].dt.year
df["date"]   = df["datetime"].dt.date
df["is_day"] = (df["H_sun"] > 0).astype(int)

# Season mapping (Indian context)
def get_season(m):
    if m in [12, 1, 2]:    return "Winter"
    elif m in [3, 4, 5]:   return "Summer"
    elif m in [6, 7, 8, 9]: return "Monsoon"
    else:                   return "Post-Monsoon"
df["season"] = df["month"].map(get_season)

# Overview
print("\ndf.info():")
print(f"  Shape       : {df.shape}")
print(f"  Date range  : {df['datetime'].iloc[0]}  →  {df['datetime'].iloc[-1]}")
print(f"  Columns     : {df.columns.tolist()}")
print(f"  Dtypes      :\n{df.dtypes}")

print("\ndf.describe():")
numeric_cols = ["P", "G_i", "Gb_i", "Gd_i", "Gr_i", "H_sun", "T2m", "WS10m", "RH", "SP", "WD10m"]
print(df[numeric_cols].describe().round(2).to_string())

print("\nMissing values:")
print(df[numeric_cols].isnull().sum().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: UNIVARIATE ANALYSIS")
print("=" * 65)

# --- 2a. Target variable P distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full dataset
sns.histplot(df["P"], bins=50, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("PV Power (P) Distribution — All Hours")
axes[0].set_xlabel("Power Output (W)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df["P"].mean(), color="red", linestyle="--", label=f"Mean = {df['P'].mean():.1f} W")
axes[0].legend()

# Daytime only
day_P = df[df["P"] > 0]["P"]
sns.histplot(day_P, bins=40, kde=True, ax=axes[1], color="orange")
axes[1].set_title("PV Power (P) Distribution — Daytime Only")
axes[1].set_xlabel("Power Output (W)")
axes[1].set_ylabel("Frequency")
axes[1].axvline(day_P.mean(), color="red", linestyle="--", label=f"Mean = {day_P.mean():.1f} W")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT / "01_P_distribution.png")
plt.close()
print("  ✓ 01_P_distribution.png")

# --- 2b. Irradiance G_i distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df[df["G_i"] > 0]["G_i"], bins=40, kde=True, color="gold")
plt.title("Global Irradiance on Inclined Plane (G_i) — Daytime")
plt.xlabel("Irradiance (W/m²)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUT / "02_Gi_distribution.png")
plt.close()
print("  ✓ 02_Gi_distribution.png")

# --- 2c. Temperature distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["T2m"], bins=40, kde=True, color="tomato")
plt.title("Air Temperature (T2m) Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.axvline(df["T2m"].mean(), color="darkred", linestyle="--", label=f"Mean = {df['T2m'].mean():.1f}°C")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "03_T2m_distribution.png")
plt.close()
print("  ✓ 03_T2m_distribution.png")

# --- 2d. Wind Speed distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["WS10m"], bins=30, kde=True, color="skyblue")
plt.title("Wind Speed (WS10m) Distribution")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUT / "04_WS10m_distribution.png")
plt.close()
print("  ✓ 04_WS10m_distribution.png")

# --- 2e. Relative Humidity distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["RH"], bins=40, kde=True, color="mediumseagreen")
plt.title("Relative Humidity (RH) Distribution")
plt.xlabel("Humidity (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUT / "05_RH_distribution.png")
plt.close()
print("  ✓ 05_RH_distribution.png")

# --- 2f. Box plots of all features ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col, color in zip(
    axes.flat,
    ["P", "G_i", "T2m", "WS10m", "RH", "SP"],
    ["steelblue", "gold", "tomato", "skyblue", "mediumseagreen", "plum"]
):
    sns.boxplot(y=df[col], ax=ax, color=color, width=0.4)
    ax.set_title(f"{col} Box Plot")
    ax.set_ylabel(col)
plt.suptitle("Box Plots of Key Features", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "06_boxplots_all.png", bbox_inches="tight")
plt.close()
print("  ✓ 06_boxplots_all.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: BIVARIATE ANALYSIS")
print("=" * 65)

day = df[df["P"] > 0].copy()

# --- 3a. G_i vs P (scatter) ---
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(day["G_i"], day["P"], c=day["T2m"], cmap="coolwarm",
                     alpha=0.4, s=8, edgecolors="none")
plt.colorbar(scatter, label="Temperature (°C)")
ax.set_xlabel("Global Irradiance G_i (W/m²)")
ax.set_ylabel("PV Power P (W)")
ax.set_title("G_i vs P — Coloured by Temperature")
plt.tight_layout()
plt.savefig(OUT / "07_Gi_vs_P_scatter.png")
plt.close()
print("  ✓ 07_Gi_vs_P_scatter.png")

# --- 3b. Temperature vs P (scatter, daytime) ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="T2m", y="P", 
                data=day, 
                alpha=0.5, s=15, color="tomato", ax=ax)
sns.regplot(x="T2m", y="P", 
            data=day, 
            scatter=False, color="darkred",
            line_kws={"linewidth": 2}, ax=ax)
ax.set_title("Temperature vs P (Daytime)\nShows Panel Efficiency Loss")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("PV Power P (W)")
plt.tight_layout()
plt.savefig(OUT / "08_T2m_vs_P_controlled.png")
plt.close()
print("  ✓ 08_T2m_vs_P_controlled.png")

# --- 3c. Wind Speed vs P ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="WS10m", y="P", data=day, alpha=0.5, s=15, color="skyblue", ax=ax)
sns.regplot(x="WS10m", y="P", data=day, scatter=False, color="navy",
            line_kws={"linewidth": 2}, ax=ax)
ax.set_title("Wind Speed vs P (Daytime)\nWind Cooling Effect")
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("PV Power P (W)")
plt.tight_layout()
plt.savefig(OUT / "09_WS10m_vs_P_controlled.png")
plt.close()
print("  ✓ 09_WS10m_vs_P_controlled.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GROUP / TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: GROUP / TEMPORAL ANALYSIS")
print("=" * 65)

# --- 4a. Diurnal Profile (Hourly Mean Power) ---
hourly_mean = df.groupby("hour")["P"].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(hourly_mean.index, hourly_mean.values, color="orange", edgecolor="darkorange")
ax.set_xlabel("Hour of Day (UTC)")
ax.set_ylabel("Mean PV Power (W)")
ax.set_title("Diurnal Profile — Average PV Power by Hour")
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f"{h:02d}" for h in range(24)])
# Add IST annotation
ax.annotate("Peak ~06 UTC\n(11:30 IST)", xy=(6, hourly_mean.iloc[6]),
            xytext=(10, hourly_mean.iloc[6] + 50),
            arrowprops=dict(arrowstyle="->", color="red"), fontsize=10, color="red")
plt.tight_layout()
plt.savefig(OUT / "10_diurnal_profile.png")
plt.close()
print("  ✓ 10_diurnal_profile.png")

print("\n  Average Power by Hour (UTC):")
for h, v in hourly_mean.items():
    print(f"    {h:02d}:00  {v:6.1f} W")

# --- 4b. Monthly Mean Power & Temperature ---
monthly = df.groupby("month").agg(
    mean_P=("P", "mean"), mean_Gi=("G_i", "mean"),
    mean_T=("T2m", "mean"), mean_RH=("RH", "mean")
).round(1)
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax1 = plt.subplots(figsize=(10, 6))
bar_width = 0.35
x = np.arange(12)
bars1 = ax1.bar(x - bar_width/2, monthly["mean_P"], bar_width, color="steelblue", label="Mean P (W)")
bars2 = ax1.bar(x + bar_width/2, monthly["mean_Gi"], bar_width, color="gold", alpha=0.7, label="Mean G_i (W/m²)")
ax1.set_xlabel("Month")
ax1.set_ylabel("W / W/m²")
ax1.set_xticks(x)
ax1.set_xticklabels(month_labels)

ax2 = ax1.twinx()
ax2.plot(x, monthly["mean_T"], color="tomato", marker="o", linewidth=2, label="Mean T2m (°C)")
ax2.plot(x, monthly["mean_RH"], color="mediumseagreen", marker="s", linewidth=2, label="Mean RH (%)")
ax2.set_ylabel("°C / %")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.title("Monthly Profile — Power, Irradiance, Temperature & Humidity")
plt.tight_layout()
plt.savefig(OUT / "11_monthly_profile.png")
plt.close()
print("  ✓ 11_monthly_profile.png")

print("\n  Monthly Summary:")
monthly.index = month_labels
print(monthly.to_string())

# --- 4c. Seasonal Box Plot ---
season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(x="season", y="P", data=df[df["P"] > 0], order=season_order,
            palette="Set2", ax=axes[0])
axes[0].set_title("PV Power by Season (Daytime Only)")
axes[0].set_ylabel("P (W)")

sns.boxplot(x="season", y="T2m", data=df, order=season_order,
            palette="Set1", ax=axes[1])
axes[1].set_title("Temperature by Season")
axes[1].set_ylabel("T2m (°C)")

plt.tight_layout()
plt.savefig(OUT / "12_seasonal_boxplots.png")
plt.close()
print("  ✓ 12_seasonal_boxplots.png")

print("\n  Mean Power by Season (daytime):")
seas = df[df["P"] > 0].groupby("season")["P"].agg(["mean","median","max","std"]).round(1)
print(seas.reindex(season_order).to_string())

# --- 4d. Year Comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
for yr, color in zip([2019, 2020], ["steelblue", "coral"]):
    yr_data = df[df["year"] == yr].groupby("month")["P"].mean()
    ax.plot(range(1, 13), yr_data.values, marker="o", linewidth=2, color=color, label=str(yr))
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.set_xlabel("Month")
ax.set_ylabel("Mean PV Power (W)")
ax.set_title("Year-over-Year Comparison (2019 vs 2020)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "13_year_comparison.png")
plt.close()
print("  ✓ 13_year_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MULTIVARIATE ANALYSIS (CORRELATION HEATMAP)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5: MULTIVARIATE ANALYSIS — CORRELATION HEATMAP")
print("=" * 65)

# --- 5a. Full Correlation Heatmap ---
corr_cols = ["P", "G_i", "Gb_i", "Gd_i", "Gr_i", "H_sun", "T2m", "WS10m", "RH", "SP", "WD10m"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", mask=mask,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix — All Hours", fontsize=15)
plt.tight_layout()
plt.savefig(OUT / "14_correlation_heatmap_all.png")
plt.close()
print("  ✓ 14_correlation_heatmap_all.png")

# --- 5b. Daytime-Only Correlation Heatmap ---
corr_day = day[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_day, dtype=bool))
sns.heatmap(corr_day, annot=True, cmap="coolwarm", fmt=".2f", mask=mask,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix — Daytime Only (P > 0)", fontsize=15)
plt.tight_layout()
plt.savefig(OUT / "15_correlation_heatmap_daytime.png")
plt.close()
print("  ✓ 15_correlation_heatmap_daytime.png")

# --- 5c. Pairplot (top features) ---
pair_cols = ["P", "G_i", "T2m", "WS10m", "RH"]
pair_data = day[pair_cols].sample(2000, random_state=42)
g = sns.pairplot(pair_data, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10},
                 height=2.2, aspect=1.1)
g.figure.suptitle("Pairplot — Key Features (Daytime Sample)", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "16_pairplot.png", bbox_inches="tight")
plt.close()
print("  ✓ 16_pairplot.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — OUTLIER & SKEWNESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6: OUTLIER & SKEWNESS ANALYSIS")
print("=" * 65)

# Skewness & Kurtosis
print("\n  Skewness & Kurtosis:")
print(f"  {'Feature':12s}  {'Skewness':>10s}  {'Kurtosis':>10s}  {'Interpretation'}")
print("  " + "-" * 60)
for col in numeric_cols:
    sk = df[col].skew()
    ku = df[col].kurtosis()
    if abs(sk) < 0.5:
        interp = "~symmetric"
    elif sk > 0:
        interp = "right-skewed (long right tail)"
    else:
        interp = "left-skewed (long left tail)"
    print(f"  {col:12s}  {sk:+10.3f}  {ku:+10.3f}  {interp}")

# --- Violin Plots for day/night split ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col in zip(axes.flat, ["T2m", "WS10m", "RH", "SP", "WD10m", "H_sun"]):
    sns.violinplot(x="is_day", y=col, data=df, ax=ax, palette=["#5688C7", "#F4A460"],
                   inner="quartile")
    ax.set_xticklabels(["Night", "Day"])
    ax.set_title(f"{col} — Night vs Day")
plt.suptitle("Feature Distributions: Night vs Day", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "17_violin_night_vs_day.png", bbox_inches="tight")
plt.close()
print("  ✓ 17_violin_night_vs_day.png")

# --- IQR-based Outlier Count ---
print("\n  IQR Outlier Detection (daytime P > 0):")
for col in ["P", "G_i", "T2m", "WS10m", "RH"]:
    q1, q3 = day[col].quantile(0.25), day[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((day[col] < lower) | (day[col] > upper)).sum()
    pct = outliers / len(day) * 100
    print(f"    {col:10s}  IQR=[{q1:.1f}, {q3:.1f}]  bounds=[{lower:.1f}, {upper:.1f}]  outliers={outliers} ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — SHAP FEATURE IMPORTANCE & DEPENDENCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7: SHAP ANALYSIS")
print("=" * 65)

# Time-based cyclic features
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

features = ["G_i", "Gb_i", "Gd_i", "Gr_i", "H_sun", "T2m", "WS10m",
            "RH", "SP", "WD10m", "hour_sin", "hour_cos", "month_sin", "month_cos"]

# ── ALL-HOURS MODEL ──
X_all, y_all = df[features].values, df["P"].values
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)

model_all = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model_all.fit(X_tr, y_tr, verbose=False)
yp_all = model_all.predict(X_te)

print(f"\n  ALL-HOURS MODEL:")
print(f"    MAE  = {mean_absolute_error(y_te, yp_all):.2f} W")
print(f"    RMSE = {np.sqrt(mean_squared_error(y_te, yp_all)):.2f} W")
print(f"    R²   = {r2_score(y_te, yp_all):.5f}")

# SHAP
explainer_all = shap.TreeExplainer(model_all)
idx_all = np.random.RandomState(42).choice(len(X_te), 2000, replace=False)
sv_all = explainer_all.shap_values(X_te[idx_all])

print("\n  SHAP Importance (All Hours):")
imp_all = pd.Series(np.abs(sv_all).mean(0), index=features).sort_values(ascending=False)
for f, v in imp_all.items():
    print(f"    {f:14s} {v:7.2f} W  {'█' * int(v / 3)}")

# Bar Plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(sv_all, X_te[idx_all], feature_names=features, plot_type="bar", show=False)
plt.title("SHAP Feature Importance — All Hours", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "18_shap_bar_all.png")
plt.close()
print("  ✓ 18_shap_bar_all.png")

# Beeswarm
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv_all, X_te[idx_all], feature_names=features, show=False)
plt.title("SHAP Beeswarm — All Hours", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "19_shap_beeswarm_all.png")
plt.close()
print("  ✓ 19_shap_beeswarm_all.png")

# ── DAYTIME-ONLY MODEL ──
day_full = df[df["H_sun"] > 0].copy()
X_day, y_day = day_full[features].values, day_full["P"].values
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_day, y_day, test_size=0.2, shuffle=False)

model_day = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model_day.fit(X_tr2, y_tr2, verbose=False)
yp_day = model_day.predict(X_te2)

print(f"\n  DAYTIME-ONLY MODEL ({len(day_full):,} rows):")
print(f"    MAE  = {mean_absolute_error(y_te2, yp_day):.2f} W")
print(f"    RMSE = {np.sqrt(mean_squared_error(y_te2, yp_day)):.2f} W")
print(f"    R²   = {r2_score(y_te2, yp_day):.5f}")

# SHAP daytime
explainer_day = shap.TreeExplainer(model_day)
idx_day = np.random.RandomState(42).choice(len(X_te2), min(2000, len(X_te2)), replace=False)
sv_day = explainer_day.shap_values(X_te2[idx_day])
X_sample_day = X_te2[idx_day]

print("\n  SHAP Importance (Daytime Only):")
imp_day = pd.Series(np.abs(sv_day).mean(0), index=features).sort_values(ascending=False)
for f, v in imp_day.items():
    print(f"    {f:14s} {v:7.2f} W  {'█' * int(v / 3)}")

# Direction of effect
print("\n  SHAP Direction of Effect (Daytime):")
for i, col in enumerate(features):
    corr = np.corrcoef(X_sample_day[:, i], sv_day[:, i])[0, 1]
    if corr > 0.1:
        d = "↑ more → higher P"
    elif corr < -0.1:
        d = "↓ more → lower P"
    else:
        d = "○ mixed/non-linear"
    print(f"    {col:14s}  corr={corr:+.3f}  {d}")

# Bar Plot (daytime)
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(sv_day, X_sample_day, feature_names=features, plot_type="bar", show=False)
plt.title("SHAP Feature Importance — Daytime Only", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "20_shap_bar_daytime.png")
plt.close()
print("  ✓ 20_shap_bar_daytime.png")

# Beeswarm (daytime)
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(sv_day, X_sample_day, feature_names=features, show=False)
plt.title("SHAP Beeswarm — Daytime Only", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "21_shap_beeswarm_daytime.png")
plt.close()
print("  ✓ 21_shap_beeswarm_daytime.png")

# Dependence Plots (top 4 features)
dep_features = ["G_i", "T2m", "H_sun", "RH"]
for feat_name in dep_features:
    fig, ax = plt.subplots(figsize=(8, 5))
    feat_idx = features.index(feat_name)
    shap.dependence_plot(feat_idx, sv_day, X_sample_day, feature_names=features,
                         show=False, ax=ax)
    plt.title(f"SHAP Dependence: {feat_name} (Daytime)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / f"22_shap_dep_{feat_name}.png")
    plt.close()
    print(f"  ✓ 22_shap_dep_{feat_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("EDA COMPLETE — All plots saved to: eda_plots/")
print("=" * 65)

plot_list = sorted(OUT.glob("*.png"))
print(f"\n  Total plots generated: {len(plot_list)}")
for p in plot_list:
    print(f"    {p.name}")

print("\nDone.")
