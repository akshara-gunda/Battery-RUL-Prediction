from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.data import feature_ranges, load_master
from src.model_io import load_metadata, load_model


ROOT = Path(__file__).resolve().parent
MASTER_PATH = ROOT / "Datasets" / "Battery_RUL_Master.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

st.set_page_config(page_title="Battery RUL Monitor", page_icon="🔋", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.65rem; white-space: nowrap; }
    @media (max-width: 640px) {
        [data-testid="stMetricValue"] { font-size: 1.25rem; }
        h1 { font-size: 2rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_master(MASTER_PATH)


@st.cache_resource
def load_artifacts(mode: str):
    metadata = load_metadata(MODEL_DIR / f"{mode}_metadata.json")
    model = load_model(ROOT / metadata["model_path"])
    return model, metadata


def risk_label(predicted_rul: float, maximum_rul: float) -> str:
    fraction = predicted_rul / maximum_rul if maximum_rul else 0
    if fraction > 0.5:
        return "Healthy"
    if fraction > 0.15:
        return "Approaching EOL"
    return "End-of-Life Risk"


def prediction_frame(values: dict, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame([values], columns=features)


def local_explanation(model, inputs: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    explanation = shap.TreeExplainer(model)(inputs)
    contributions = explanation.values[0]
    return pd.DataFrame(
        {
            "feature": features,
            "contribution": contributions,
            "absolute_contribution": abs(contributions),
            "value": inputs.iloc[0].to_numpy(),
        }
    ).sort_values("absolute_contribution", ascending=False)


data = load_data()
maximum_rul = float(data["RUL"].max())

st.title("Battery RUL Monitor")
st.caption("Compare cycle-aware and telemetry-only remaining-life predictions")

mode_labels = {
    "telemetry_only": "Telemetry-only",
    "cycle_aware": "Cycle-aware",
}
mode = st.sidebar.radio(
    "Prediction mode",
    list(mode_labels),
    format_func=lambda value: mode_labels[value],
)
model, metadata = load_artifacts(mode)
features = metadata["features"]
ranges = feature_ranges(data, features)

holdout_metrics = pd.read_csv(REPORT_DIR / "holdout_metrics.csv")
selected_metrics = holdout_metrics[
    (holdout_metrics["mode"] == mode) & (holdout_metrics["model"] == metadata["model"])
].iloc[0]

metric_one, metric_two = st.columns(2)
metric_one.metric("Model", metadata["model"].replace("_", " ").title())
metric_two.metric("Holdout MAE", f"{selected_metrics['mae']:.2f} cycles")
metric_three, metric_four = st.columns(2)
metric_three.metric("Holdout RMSE", f"{selected_metrics['rmse']:.2f} cycles")
metric_four.metric("Holdout R²", f"{selected_metrics['r2']:.4f}")

if mode == "telemetry_only":
    st.info("This mode uses only electrical and timing measurements. It does not use Cycle_Index.")
else:
    st.info("This mode includes Cycle_Index and is an upper-bound benchmark because RUL is strongly tied to cycle position.")

overview, predict, explain = st.tabs(["Overview", "Predict RUL", "Explainability"])

with overview:
    st.subheader("Model comparison")
    comparison = holdout_metrics[holdout_metrics["mode"] == mode]
    st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)

    predictions_path = REPORT_DIR / "cv_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        predictions = predictions[
            (predictions["mode"] == mode) & (predictions["model"] == metadata["model"])
        ]
        chart = px.scatter(
            predictions,
            x="actual_rul",
            y="predicted_rul",
            hover_data=["fold"],
            labels={"actual_rul": "Actual RUL", "predicted_rul": "Predicted RUL"},
            title=f"{mode_labels[mode]} grouped-CV predictions",
        )
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("Dataset")
    dataset_one, dataset_two, dataset_three = st.columns(3)
    dataset_one.metric("Cycle records", f"{len(data):,}")
    dataset_two.metric("Physical batteries", data["battery_id"].nunique())
    dataset_three.metric("Maximum recorded RUL", f"{maximum_rul:.0f} cycles")

with predict:
    st.subheader(f"{mode_labels[mode]} prediction")
    values = {}
    defaults = ranges["median"]
    columns = st.columns(2)
    for index, feature in enumerate(features):
        lower = float(ranges.loc[feature, "minimum"])
        upper = float(ranges.loc[feature, "maximum"])
        default = float(defaults[feature])
        with columns[index % 2]:
            values[feature] = st.number_input(
                feature,
                min_value=lower,
                max_value=upper,
                value=min(max(default, lower), upper),
                key=f"{mode}_{feature}",
            )

    if st.button("Predict RUL", type="primary"):
        inputs = prediction_frame(values, features)
        prediction = float(model.predict(inputs)[0])
        st.metric("Predicted remaining useful life", f"{prediction:.1f} cycles")
        st.info(risk_label(prediction, maximum_rul))
        st.session_state[f"inputs_{mode}"] = inputs

    st.subheader("CSV prediction")
    uploaded = st.file_uploader(
        f"Upload a CSV containing the {mode_labels[mode].lower()} features",
        type="csv",
        key=f"upload_{mode}",
    )
    if uploaded is not None:
        uploaded_data = pd.read_csv(uploaded)
        missing = sorted(set(features) - set(uploaded_data.columns))
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            predictions = model.predict(uploaded_data[features])
            result = uploaded_data.copy()
            result["predicted_RUL"] = predictions.round(2)
            result["risk"] = [risk_label(value, maximum_rul) for value in predictions]
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download predictions",
                result.to_csv(index=False),
                "battery_rul_predictions.csv",
                "text/csv",
            )

with explain:
    st.subheader("Global feature importance")
    importance_path = REPORT_DIR / f"shap_feature_importance_{mode}.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        chart = px.bar(
            importance.sort_values("mean_abs_shap"),
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title=f"Mean absolute SHAP contribution: {mode_labels[mode]}",
        )
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("Local explanation")
    inputs = st.session_state.get(f"inputs_{mode}")
    if inputs is None:
        inputs = prediction_frame(ranges["median"].to_dict(), features)
        st.caption("Showing the median feature profile.")
    contributions = local_explanation(model, inputs, features)
    chart = px.bar(
        contributions.sort_values("contribution"),
        x="contribution",
        y="feature",
        orientation="h",
        color="contribution",
        color_continuous_scale="RdBu",
        title="Feature contributions for this prediction",
    )
    st.plotly_chart(chart, use_container_width=True)
    st.dataframe(contributions.round(4), use_container_width=True, hide_index=True)
