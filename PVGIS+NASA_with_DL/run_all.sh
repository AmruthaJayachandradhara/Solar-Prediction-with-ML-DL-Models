#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/Users/amruthaj/Desktop/Desktop/Capstone"
PYTHON_BIN="$BASE_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python interpreter not found at $PYTHON_BIN"
  echo "Create or activate the project virtual environment first."
  exit 1
fi

cd "$BASE_DIR"

echo "============================================================"
echo "Solar Pipeline: download -> cleaning -> eda -> preprocessing -> model"
echo "============================================================"

echo "[1/5] Running data download..."
"$PYTHON_BIN" data/download_raw_data.py

echo "[2/5] Running cleaning..."
"$PYTHON_BIN" datacleaning/clean_data.py

echo "[3/5] Running EDA..."
"$PYTHON_BIN" EDA/eda_complete.py

echo "[4/5] Running preprocessing..."
"$PYTHON_BIN" preprocessing/scaling.py

echo "[5/5] Running model training..."
"$PYTHON_BIN" model/dual_stream.py

echo "============================================================"
echo "Pipeline complete."
echo "Outputs:"
echo "- Raw data:       $BASE_DIR/data"
echo "- Processed data: $BASE_DIR/datacleaning/processed_solar_2019_2020.csv"
echo "- EDA plots:      $BASE_DIR/EDA/EDA_plots"
echo "- Prep outputs:   $BASE_DIR/preprocessing/preprocessed_data"
echo "============================================================"
