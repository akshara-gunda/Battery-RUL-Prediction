from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .preprocess import FEATURE_COLUMNS, TARGET_COLUMN
except ImportError:
    from preprocess import FEATURE_COLUMNS, TARGET_COLUMN

TELEMETRY_FEATURES = FEATURE_COLUMNS
CYCLE_AWARE_FEATURES = ["Cycle_Index", *FEATURE_COLUMNS]
FEATURE_SETS = {
    "telemetry_only": TELEMETRY_FEATURES,
    "cycle_aware": CYCLE_AWARE_FEATURES,
}
MODEL_FEATURES = CYCLE_AWARE_FEATURES
MASTER_COLUMNS = ["Cycle_Index", "battery_id", *FEATURE_COLUMNS, TARGET_COLUMN]


def load_master(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    validate_master(data)
    return data


def validate_master(data: pd.DataFrame) -> None:
    if data.columns.tolist() != MASTER_COLUMNS:
        raise ValueError(f"Unexpected columns. Expected: {MASTER_COLUMNS}")

    numeric_columns = ["Cycle_Index", *FEATURE_COLUMNS, TARGET_COLUMN]
    if data[numeric_columns].isna().any().any():
        raise ValueError("The master dataset contains missing numeric values")
    if data.duplicated(["battery_id", "Cycle_Index"]).any():
        raise ValueError("Duplicate battery-cycle rows found")
    if (data["Cycle_Index"] < 0).any() or (data[TARGET_COLUMN] < 0).any():
        raise ValueError("Cycle_Index and RUL must be non-negative")

    final_cycles = data.groupby("battery_id")["Cycle_Index"].transform("max")
    expected_rul = final_cycles - data["Cycle_Index"]
    if not np.isclose(data[TARGET_COLUMN], expected_rul).all():
        raise ValueError("RUL is inconsistent with the final cycle for a battery")


def split_holdout(
    data: pd.DataFrame,
    holdout_battery: str = "HNEI_t",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_battery not in set(data["battery_id"]):
        raise ValueError(f"Unknown holdout battery: {holdout_battery}")

    holdout = data[data["battery_id"] == holdout_battery].copy()
    development = data[data["battery_id"] != holdout_battery].copy()
    return development, holdout


def feature_ranges(
    data: pd.DataFrame,
    features: list[str] = MODEL_FEATURES,
) -> pd.DataFrame:
    ranges = data[features].agg(["min", "max", "median"]).T
    return ranges.rename(columns={"min": "minimum", "max": "maximum"})


def anomaly_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in FEATURE_COLUMNS:
        values = data[column]
        lower = values.quantile(0.01)
        upper = values.quantile(0.99)
        rows.append(
            {
                "feature": column,
                "negative_values": int((values < 0).sum()),
                "below_1st_percentile": int((values < lower).sum()),
                "above_99th_percentile": int((values > upper).sum()),
            }
        )
    return pd.DataFrame(rows)
