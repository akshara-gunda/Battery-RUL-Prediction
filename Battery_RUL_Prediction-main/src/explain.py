from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import shap

from data import load_master
from model_io import load_metadata, load_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_explanations(
    master_path: Path,
    report_dir: Path,
    mode: str,
    local_row: int = 0,
) -> None:
    data = load_master(master_path)
    metadata = load_metadata(Path("models") / f"{mode}_metadata.json")
    model_path = Path(metadata["model_path"])
    features = metadata["features"]
    model = load_model(model_path)
    values = data[features]
    sample = values.sample(min(2000, len(values)), random_state=42)
    explanation = shap.TreeExplainer(model)(sample)

    importance = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": abs(explanation.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(report_dir / f"shap_feature_importance_{mode}.csv", index=False)

    shap.summary_plot(explanation.values, sample, show=False)
    plt.tight_layout()
    plt.savefig(report_dir / f"shap_summary_{mode}.png", dpi=180, bbox_inches="tight")
    plt.close()

    local_row = max(0, min(local_row, len(data) - 1))
    local_explanation = shap.TreeExplainer(model)(values.iloc[[local_row]])[0]
    shap.plots.waterfall(local_explanation, show=False)
    plt.tight_layout()
    plt.savefig(report_dir / f"shap_local_waterfall_{mode}.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP reports to {report_dir}")
    print(importance.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SHAP explanations for the RUL model")
    parser.add_argument("--master", type=Path, default=Path("Datasets/Battery_RUL_Master.csv"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--mode", choices=["telemetry_only", "cycle_aware", "both"], default="both")
    parser.add_argument("--local-row", type=int, default=0)
    args = parser.parse_args()
    modes = ["telemetry_only", "cycle_aware"] if args.mode == "both" else [args.mode]
    for mode in modes:
        build_explanations(args.master, args.reports, mode, args.local_row)


if __name__ == "__main__":
    main()
