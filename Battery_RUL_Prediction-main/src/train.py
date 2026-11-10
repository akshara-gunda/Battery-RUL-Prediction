from __future__ import annotations

import argparse
import hashlib
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from data import FEATURE_SETS, load_master, split_holdout
from evaluate import regression_metrics
from model_io import save_metadata, save_model


LIGHTGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "verbosity": -1,
}

MODEL_PATHS = {
    "telemetry_only": Path("models/best_telemetry_model.pkl"),
    "cycle_aware": Path("models/best_cycle_model.pkl"),
}


def make_models() -> dict:
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(**LIGHTGBM_PARAMS),
    }


def run_grouped_cv(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = data[features]
    y = data["RUL"]
    groups = data["battery_id"]
    splitter = GroupKFold(n_splits=5)
    predictions = []
    metrics = []

    for model_name in make_models():
        for fold, (train_index, valid_index) in enumerate(
            splitter.split(X, y, groups=groups), start=1
        ):
            train_groups = set(groups.iloc[train_index])
            valid_groups = set(groups.iloc[valid_index])
            if train_groups.intersection(valid_groups):
                raise RuntimeError("A battery appears in both training and validation")

            model = make_models()[model_name]
            model.fit(X.iloc[train_index], y.iloc[train_index])
            start = time.perf_counter()
            prediction = model.predict(X.iloc[valid_index])
            latency = (time.perf_counter() - start) * 1000 / len(valid_index)
            metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "validation_batteries": ", ".join(sorted(valid_groups)),
                    **regression_metrics(y.iloc[valid_index], prediction),
                    "inference_ms_per_sample": latency,
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "row_index": valid_index,
                        "model": model_name,
                        "actual_rul": y.iloc[valid_index].to_numpy(),
                        "predicted_rul": prediction,
                        "fold": fold,
                    }
                )
            )

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def run_holdout(
    data: pd.DataFrame,
    features: list[str],
    holdout_battery: str,
) -> tuple[pd.DataFrame, dict]:
    development, holdout = split_holdout(data, holdout_battery)
    results = []
    fitted_models = {}

    for model_name, model in make_models().items():
        model.fit(development[features], development["RUL"])
        prediction = model.predict(holdout[features])
        results.append(
            {
                "model": model_name,
                "holdout_battery": holdout_battery,
                **regression_metrics(holdout["RUL"], prediction),
            }
        )
        fitted_models[model_name] = model

    return pd.DataFrame(results), fitted_models


def dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_variant(
    data: pd.DataFrame,
    master_path: Path,
    report_dir: Path,
    holdout_battery: str,
    mode: str,
) -> None:
    features = list(FEATURE_SETS[mode])
    development, holdout = split_holdout(data, holdout_battery)
    cv_metrics, cv_predictions = run_grouped_cv(development, features)
    holdout_metrics, _ = run_holdout(data, features, holdout_battery)
    selected_model_name = cv_metrics.groupby("model")["mae"].mean().idxmin()

    final_model = make_models()[selected_model_name]
    final_model.fit(data[features], data["RUL"])
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_PATHS[mode]

    cv_metrics.insert(0, "mode", mode)
    cv_predictions.insert(0, "mode", mode)
    holdout_metrics.insert(0, "mode", mode)
    for report_name, report in [
        ("cv_metrics.csv", cv_metrics),
        ("cv_predictions.csv", cv_predictions),
        ("holdout_metrics.csv", holdout_metrics),
    ]:
        report_path = report_dir / report_name
        report.to_csv(
            report_path,
            mode="a",
            index=False,
            header=not report_path.exists(),
        )

    save_model(final_model, model_path)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "mode": mode,
        "model": selected_model_name,
        "model_path": str(model_path),
        "features": features,
        "target": "RUL",
        "holdout_battery": holdout_battery,
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "lightgbm_parameters": LIGHTGBM_PARAMS,
        "dataset_sha256": dataset_hash(master_path),
        "cv_mean_metrics": cv_metrics[cv_metrics["model"] == selected_model_name]
        [["mae", "rmse", "r2"]]
        .mean()
        .to_dict(),
        "holdout_metrics": holdout_metrics.to_dict(orient="records"),
    }
    save_metadata(metadata, model_path.with_name(f"{mode}_metadata.json"))

    print(f"\n{mode}: {selected_model_name}")
    print(cv_metrics.groupby("model")[['mae', 'rmse', 'r2']].mean().round(4))
    print(holdout_metrics.round(4).to_string(index=False))


def train_project(master_path: Path, report_dir: Path, holdout_battery: str) -> None:
    data = load_master(master_path)
    for report_name in ["cv_metrics.csv", "cv_predictions.csv", "holdout_metrics.csv"]:
        report_path = report_dir / report_name
        if report_path.exists():
            report_path.unlink()

    for mode in FEATURE_SETS:
        train_variant(data, master_path, report_dir, holdout_battery, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train both battery RUL feature variants")
    parser.add_argument("--master", type=Path, default=Path("Datasets/Battery_RUL_Master.csv"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--holdout-battery", default="HNEI_t")
    args = parser.parse_args()
    train_project(args.master, args.reports, args.holdout_battery)


if __name__ == "__main__":
    main()
