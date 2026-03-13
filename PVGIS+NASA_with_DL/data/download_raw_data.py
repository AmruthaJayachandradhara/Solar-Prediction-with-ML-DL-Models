"""
Download raw data from NASA POWER and PVGIS APIs
Location: Rajasthan, India
Coordinates: lat=26.810578, lon=73.768455
"""

import requests
import json
import csv
import os
import time

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
LAT = 26.810578
LON = 73.768455
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PVGIS_BASE  = "https://re.jrc.ec.europa.eu/api/v5_2"
NASA_BASE   = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def save_json(data: dict, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [saved] {filename}")


def save_csv(rows: list[dict], filename: str) -> None:
    if not rows:
        print(f"  [WARN]  No rows to write for {filename}")
        return
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [saved] {filename}  ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════
#  PVGIS  ─  Hourly series 2019-2020
# ═══════════════════════════════════════════════════════════════
def download_pvgis_hourly() -> None:
    print("\n── PVGIS Hourly 2019-2020 ──────────────────────────────")

    params = {
        "lat":          LAT,
        "lon":          LON,
        "startyear":    2019,
        "endyear":      2020,
        "pvcalculation": 1,       # include P (PV output)
        "peakpower":    1,         # 1 kWp system
        "loss":         14,        # 14 % system losses
        "angle":        30,        # 30° tilt
        "aspect":       0,         # south-facing (0 = south)
        "components":   1,         # return Gb(i), Gd(i), Gr(i)
        "usehorizon":   1,
        "outputformat": "json",
    }

    url = f"{PVGIS_BASE}/seriescalc"
    print(f"  GET {url}")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    # ── raw JSON ──
    save_json(data, "raw_pvgis_2019_2020.json")

    # ── metadata ──
    meta = {
        "url":    r.url,
        "params": params,
        "inputs": data.get("inputs", {}),
        "meta":   data.get("meta",   {}),
    }
    save_json(meta, "raw_pvgis_metadata.json")

    # ── hourly CSV ──
    hourly = data.get("outputs", {}).get("hourly", [])
    save_csv(hourly, "raw_pvgis_hourly_2019_2020.csv")


# ═══════════════════════════════════════════════════════════════
#  PVGIS  ─  Typical Meteorological Year (TMY)
# ═══════════════════════════════════════════════════════════════
def download_pvgis_tmy() -> None:
    print("\n── PVGIS TMY ───────────────────────────────────────────")

    params = {
        "lat":          LAT,
        "lon":          LON,
        "outputformat": "json",
        "usehorizon":   1,
    }

    url = f"{PVGIS_BASE}/tmy"
    print(f"  GET {url}")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    # ── raw JSON ──
    save_json(data, "raw_pvgis_tmy.json")

    # ── hourly CSV ──
    hourly = data.get("outputs", {}).get("tmy_hourly", [])
    save_csv(hourly, "raw_pvgis_tmy.csv")


# ═══════════════════════════════════════════════════════════════
#  NASA POWER  ─  Hourly (one year at a time)
# ═══════════════════════════════════════════════════════════════
NASA_PARAMS = "T2M,RH2M,WS10M,PS,ALLSKY_SFC_SW_DWN"

def download_nasa_year(year: int) -> None:
    print(f"\n── NASA POWER {year} ─────────────────────────────────────")

    params = {
        "parameters": NASA_PARAMS,
        "community":  "RE",
        "longitude":  LON,
        "latitude":   LAT,
        "start":      f"{year}0101",
        "end":        f"{year}1231",
        "format":     "JSON",
    }

    print(f"  GET {NASA_BASE}")
    r = requests.get(NASA_BASE, params=params, timeout=180)
    r.raise_for_status()
    data = r.json()

    # ── raw JSON ──
    save_json(data, f"raw_nasa_{year}.json")

    # ── flatten to rows ──
    param_data = (
        data.get("properties", {})
            .get("parameter", {})
    )

    if not param_data:
        print(f"  [WARN]  No parameter data found in NASA {year} response")
        return

    # timestamps are keys in each parameter dict (YYYYMMDDHH)
    timestamps = sorted(next(iter(param_data.values())).keys())
    rows = []
    for ts in timestamps:
        row = {"time": ts}
        for pname, pvals in param_data.items():
            row[pname] = pvals.get(ts, "")
        rows.append(row)

    save_csv(rows, f"raw_nasa_{year}.csv")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Downloading raw solar/weather data")
    print(f"  Location: lat={LAT}, lon={LON}  (Rajasthan, India)")
    print("=" * 60)

    try:
        download_pvgis_hourly()
    except Exception as e:
        print(f"  [ERROR] PVGIS hourly failed: {e}")

    time.sleep(2)   # be polite to the API

    try:
        download_pvgis_tmy()
    except Exception as e:
        print(f"  [ERROR] PVGIS TMY failed: {e}")

    time.sleep(2)

    try:
        download_nasa_year(2019)
    except Exception as e:
        print(f"  [ERROR] NASA 2019 failed: {e}")

    time.sleep(2)

    try:
        download_nasa_year(2020)
    except Exception as e:
        print(f"  [ERROR] NASA 2020 failed: {e}")

    print("\n" + "=" * 60)
    print("  Done. Check files in:", OUTPUT_DIR)
    print("=" * 60)
