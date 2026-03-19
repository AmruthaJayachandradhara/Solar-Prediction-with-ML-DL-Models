from pathlib import Path

import pandas as pd


def add_daytime_flag(
	input_csv: Path,
	output_csv: Path,
	sun_col: str = "H_sun",
	flag_col: str = "is_daytime",
) -> None:
	df = pd.read_csv(input_csv)

	if sun_col not in df.columns:
		raise ValueError(f"Column '{sun_col}' not found in {input_csv}")

	# Daytime is when the solar elevation proxy is above horizon.
	df[flag_col] = (pd.to_numeric(df[sun_col], errors="coerce").fillna(0) > 0).astype(int)

	output_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_csv, index=False)
	print(f"Saved: {output_csv}")
	print(df[[sun_col, flag_col]].head())


if __name__ == "__main__":
	base = Path(__file__).resolve().parents[1]
	input_file = base / "datacleaning" / "processed_solar_2019_2020.csv"
	output_file = base / "preprocessing" / "processed_solar_2019_2020_with_dayflag.csv"

	add_daytime_flag(input_file, output_file)
