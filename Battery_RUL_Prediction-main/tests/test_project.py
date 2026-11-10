from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import CYCLE_AWARE_FEATURES, FEATURE_SETS, load_master, split_holdout
from evaluate import regression_metrics
from model_io import load_metadata, load_model


ROOT = Path(__file__).resolve().parents[1]


def test_master_dataset_contract():
    data = load_master(ROOT / "Datasets" / "Battery_RUL_Master.csv")
    assert len(data) == 15064
    assert data["battery_id"].nunique() == 14
    assert not data.duplicated(["battery_id", "Cycle_Index"]).any()


def test_holdout_contains_a_complete_unseen_battery():
    data = load_master(ROOT / "Datasets" / "Battery_RUL_Master.csv")
    development, holdout = split_holdout(data, "HNEI_t")
    assert set(development["battery_id"]).isdisjoint(holdout["battery_id"])
    assert holdout["battery_id"].nunique() == 1
    assert list(development[CYCLE_AWARE_FEATURES].columns) == CYCLE_AWARE_FEATURES


def test_regression_metrics_are_zero_for_perfect_predictions():
    actual = pd.Series([1, 4, 8])
    scores = regression_metrics(actual, actual)
    assert scores == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_saved_models_match_metadata():
    for mode, features in FEATURE_SETS.items():
        metadata = load_metadata(ROOT / "models" / f"{mode}_metadata.json")
        model = load_model(ROOT / metadata["model_path"])
        assert metadata["features"] == features
        assert hasattr(model, "predict")
