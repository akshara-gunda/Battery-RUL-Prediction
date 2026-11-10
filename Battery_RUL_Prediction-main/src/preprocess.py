"""Combine the processed per-battery HNEI tables into one master CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "Discharge Time (s)",
    "Decrement 3.6-3.4V (s)",
    "Max. Voltage Dischar. (V)",
    "Min. Voltage Charg. (V)",
    "Time at 4.15V (s)",
    "Time constant current (s)",
    "Charging time (s)",
]
TARGET_COLUMN = "RUL"
IDENTIFIER_COLUMNS = ["Cycle_Index", "battery_id"]
MASTER_COLUMNS = IDENTIFIER_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]


def assemble_master_dataset(
    processed_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Combine the processed battery tables into one master dataset."""
    files = sorted(processed_dir.glob("HNEI_*_features.csv"))
    if len(files) != 14:
        raise ValueError(f"Expected 14 processed battery files, found {len(files)}")

    tables = []
    for path in files:
        battery_code = path.stem.removeprefix("HNEI_").removesuffix("_features")
        table = pd.read_csv(path)
        table = table.loc[:, ~table.columns.astype(str).str.startswith("Unnamed")]
        required_columns = {"Cycle_Index", TARGET_COLUMN, *FEATURE_COLUMNS}
        missing_columns = sorted(required_columns - set(table.columns))
        if missing_columns:
            raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

        table = table.loc[:, ["Cycle_Index", *FEATURE_COLUMNS, TARGET_COLUMN]]
        table.insert(1, "battery_id", f"HNEI_{battery_code}")
        tables.append(table)

    master = pd.concat(tables, ignore_index=True)
    master = master.sort_values(["battery_id", "Cycle_Index"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("Datasets/HNEI_Processed"))
    parser.add_argument("--output", type=Path, default=Path("Datasets/Battery_RUL_Master.csv"))
    args = parser.parse_args()
    master = assemble_master_dataset(args.processed_dir, args.output)
    print(f"Saved {len(master):,} rows for {master['battery_id'].nunique()} batteries to {args.output}")


if __name__ == "__main__":
    main()