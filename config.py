"""
Central path configuration for Solar PV Prediction project.
All scripts import paths from here — no hardcoded paths anywhere else.

Data folder layout (Google Drive or local):
  data/
  ├── raw/            ← download_raw_data.py output
  └── processed/      ← cleaning/clean_data.py output

Usage:
  from config import RAW_DIR, PROCESSED_DIR, EDA_PLOTS_DIR
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# PROJECT ROOT  (this repo's top-level directory)
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────
# DATA ROOT
# ──────────────────────────────────────────────────────────────
# Option 1 (default): "data/" folder inside the project
# Option 2: Set the env var  SOLAR_DATA_DIR  to point elsewhere,
#            e.g. your Google Drive sync folder.
#
# Google Drive link (read-only reference):
#   https://drive.google.com/drive/folders/1lFXGtXVp5EW3QkfoZwjuRlKZXNLLt_rP
# ──────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("SOLAR_DATA_DIR", PROJECT_ROOT / "data"))

# ── Subdirectories ────────────────────────────────────────────
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_PLOTS_DIR = PROJECT_ROOT / "EDA" / "plots"

# ── Key file paths used across scripts ────────────────────────
# Raw files produced by download_raw_data.py
RAW_PVGIS_HOURLY_CSV = RAW_DIR / "raw_pvgis_hourly_2019_2020.csv"
RAW_PVGIS_TMY_CSV    = RAW_DIR / "raw_pvgis_tmy.csv"
RAW_NASA_2019_CSV    = RAW_DIR / "raw_nasa_2019.csv"
RAW_NASA_2020_CSV    = RAW_DIR / "raw_nasa_2020.csv"

# Processed file produced by cleaning/clean_data.py
PROCESSED_CSV   = PROCESSED_DIR / "processed_solar_2019_2020.csv"
CLEANING_REPORT = PROCESSED_DIR / "cleaning_report.txt"


def ensure_dirs() -> None:
    """Create all required directories if they don't exist."""
    for d in [RAW_DIR, PROCESSED_DIR, EDA_PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
