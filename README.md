# Solar Prediction with ML/DL Models

Solar Energy Production Prediction Using Advanced Machine Learning / Deep Learning.

**Location:** Rajasthan, India (lat=26.81, lon=73.77)
**Data sources:** PVGIS hourly + TMY, NASA POWER

---

## Project Structure

```
Solar-Prediction-with-ML-DL-Models/
├── config.py                  # Central path configuration (edit DATA_DIR here)
├── download_raw_data.py       # Step 1 — Download raw data from PVGIS & NASA APIs
├── cleaning/
│   └── clean_data.py          # Step 2 — Clean, merge & export processed dataset
├── EDA/
│   └── eda_complete.py        # Step 3 — Exploratory Data Analysis (15+ plots)
│   └── plots/                 # EDA output plots (auto-generated)
├── data/
│   ├── raw/                   # Raw API downloads (CSV + JSON)
│   └── processed/             # Cleaned dataset + report
├── .gitignore
└── README.md
```

## Data

All data lives in `data/` (default) or a custom folder.

**Google Drive (raw + processed):**
<https://drive.google.com/drive/folders/1lFXGtXVp5EW3QkfoZwjuRlKZXNLLt_rP>

To point scripts at a different data folder (e.g. your Google Drive sync), set the
environment variable before running any script:

```bash
export SOLAR_DATA_DIR="/path/to/your/google-drive/data-folder"
```

Or edit `DATA_DIR` directly in `config.py`.

## Pipeline

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `download_raw_data.py` | PVGIS & NASA APIs | `data/raw/*.csv`, `data/raw/*.json` |
| 2 | `cleaning/clean_data.py` | `data/raw/*.csv` | `data/processed/processed_solar_2019_2020.csv` |
| 3 | `EDA/eda_complete.py` | `data/processed/processed_solar_2019_2020.csv` | `EDA/plots/*.png` |

Run in order:

```bash
python download_raw_data.py
python cleaning/clean_data.py
python EDA/eda_complete.py
```

## Requirements

- Python 3.10+
- pandas, numpy, requests, matplotlib, seaborn, xgboost, shap, scikit-learn
