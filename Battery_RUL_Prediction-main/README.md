# Battery RUL Prediction

This project estimates the remaining useful life (RUL) of lithium-ion batteries from cycle-level voltage and timing measurements.

The repository currently covers:

- Combining 14 processed HNEI battery files.
- Validating and documenting the resulting dataset.
- Training cycle-aware and telemetry-only model variants.
- Selecting the best model with grouped cross-validation.
- Explaining the selected tree model with SHAP.
- Serving predictions through a Streamlit dashboard.

RUL is measured in cycles. For a row at cycle `c`, the target is the number of cycles remaining until that battery reaches its recorded final cycle.

## What is the use of this project?

The project is an early battery-health monitoring system. It takes measurements from a battery cycle and estimates how many cycles remain before the battery reaches its recorded end of life. This can help an operator compare battery condition, identify batteries approaching end of life, and plan maintenance or replacement before failure.

The dashboard demonstrates the complete workflow from measured telemetry to an RUL estimate and an explanation of which features influenced that estimate. It is a course-project prototype, not a production safety system.

## Dataset

The source files are already processed cycle-level tables from the HNEI battery dataset. Each row represents one cycle from one physical battery.

The final file contains 15,064 rows from 14 batteries:

```text
Datasets/Battery_RUL_Master.csv
```

Columns:

| Column | Role |
| --- | --- |
| `Cycle_Index` | Current battery cycle number |
| `battery_id` | Physical battery identifier used for grouping |
| `Discharge Time (s)` | Discharge duration |
| `Decrement 3.6-3.4V (s)` | Time through the 3.6 V to 3.4 V discharge range |
| `Max. Voltage Dischar. (V)` | Maximum discharge voltage |
| `Min. Voltage Charg. (V)` | Minimum charging voltage |
| `Time at 4.15V (s)` | Time near the upper charging voltage |
| `Time constant current (s)` | Constant-current charging duration |
| `Charging time (s)` | Total charging duration |
| `RUL` | Prediction target in remaining cycles |

`battery_id` is retained for validation but is not used as a model feature.

## Project structure

```text
Datasets/HNEI_Processed/              input table for each battery
Datasets/Battery_RUL_Master.csv       combined dataset
src/preprocess.py                     merge-only script
src/data.py                           dataset contract and validation
src/train.py                          grouped CV, holdout, and model export
src/evaluate.py                       regression metrics
src/model_io.py                       model and metadata loading/saving
src/explain.py                        SHAP report generation
notebooks/01_prepare_data.ipynb       inspection and data validation
notebooks/02_lightgbm_training.ipynb  model training and evaluation
app.py                                Streamlit dashboard
models/best_cycle_model.pkl           selected cycle-aware model
models/best_telemetry_model.pkl       selected telemetry-only model
models/*_metadata.json                model contracts and metrics
reports/                              evaluation and SHAP outputs
tests/test_project.py                 core regression tests
docs/reviewer_notes.md                short presentation and viva guide
```

## Workflow

### 1. Combine the data

`src/preprocess.py` reads the 14 files in `Datasets/HNEI_Processed/`, removes unwanted unnamed index columns, keeps the required columns, adds `battery_id`, sorts the rows, and writes the master CSV.

### 2. Prepare and inspect the data

`01_prepare_data.ipynb` checks the source files and the master dataset. It verifies the schema, data types, missing values, duplicate battery-cycle rows, cycle ranges, and basic feature distributions.

### 3. Train and select the model

`src/train.py` trains two feature variants. The cycle-aware variant uses `Cycle_Index` and the seven telemetry columns. The telemetry-only variant uses only the seven telemetry columns. Both compare a mean baseline, Random Forest, and LightGBM. The selected model for each variant is the one with the lowest grouped-CV MAE.

LightGBM is retained because it is a strong nonlinear tabular model, but the current data selects Random Forest for both variants on the documented validation setup.

### 4. Validate without leakage

The development data uses five-fold `GroupKFold` with `battery_id` as the group. Complete batteries are held out for validation, so cycles from one physical battery cannot appear in both training and validation. `HNEI_t` is kept as an untouched final holdout.

This is more realistic than randomly splitting rows because cycles from the same battery are closely related.

### 5. Explain and use the model

`src/explain.py` creates separate global SHAP feature importance and local waterfall reports for both selected tree models. SHAP describes how the model uses the features; it does not prove causal battery behavior.

`app.py` provides:

- A switch between cycle-aware and telemetry-only models.
- Model and holdout metrics for the selected mode.
- Manual single-cycle prediction.
- CSV batch prediction.
- Actual-vs-predicted validation plots.
- Global and local SHAP charts.

## Current results

The current grouped validation report is:

```text
Cycle-aware Random Forest:
Grouped-CV MAE: 5.80 cycles
Grouped-CV RMSE: 7.94 cycles
Grouped-CV R2: 0.9992
Holdout MAE: 3.24 cycles
Holdout RMSE: 4.07 cycles
Holdout R2: 0.9998

Telemetry-only Random Forest:
Grouped-CV MAE: 30.94 cycles
Grouped-CV RMSE: 46.96 cycles
Grouped-CV R2: 0.9779
Holdout MAE: 73.90 cycles
Holdout RMSE: 92.69 cycles
Holdout R2: 0.9172

Holdout battery: `HNEI_t`
```

MAE is the average absolute prediction error. RMSE gives more weight to larger errors. R2 measures how much variation in RUL is explained by the model.

The holdout is a complete battery that was not used for model selection. These results should not be interpreted as proof that the model will perform equally well on batteries from a different manufacturer, chemistry, or testing setup.

## Run the project

Run these commands from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 src/preprocess.py
jupyter notebook notebooks/01_prepare_data.ipynb
jupyter notebook notebooks/02_lightgbm_training.ipynb
PYTHONPATH=src python3 src/train.py
PYTHONPATH=src python3 src/explain.py --mode both
streamlit run app.py
pytest -q
```

The notebooks can also be opened and run directly in VS Code with the Jupyter extension.

## Future completion

The next useful additions are:

1. Validate the selected model on batteries from another source or chemistry.
2. Add uncertainty intervals for dashboard predictions.
3. Tune the selected model using only the development batteries.
4. Add experiment tracking and automated CI checks.
5. Package the dashboard for deployment after the data contract is stable.
