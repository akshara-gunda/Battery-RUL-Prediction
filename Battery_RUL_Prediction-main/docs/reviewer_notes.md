# Reviewer Notes

## One-minute explanation

This project estimates the remaining useful life of lithium-ion batteries from cycle-level voltage and charging/discharging measurements. The intended use is battery-health monitoring: estimate the cycles remaining, identify batteries approaching end of life, and support maintenance or replacement planning.

The data contains 14 physical HNEI battery cells and 15,064 cycle records. I combined the processed files, added `battery_id`, checked the schema and data quality, and saved one master dataset.

I trained two versions of the model. The cycle-aware version uses `Cycle_Index` and the seven telemetry features. The telemetry-only version uses only the seven electrical and timing features. This comparison is important because `RUL` is calculated from final cycle minus current cycle, so `Cycle_Index` gives the model a strong shortcut. I therefore present the telemetry-only version as the more meaningful health-signal experiment and the cycle-aware version as an upper-bound benchmark.

Both variants use a fixed complete-battery holdout and five-fold `GroupKFold` validation. Grouping by `battery_id` prevents cycles from the same physical battery appearing in both training and validation.

The current selected algorithm for both modes is Random Forest. LightGBM is also trained as a nonlinear tabular benchmark. SHAP explanations show which input features influenced a prediction, while the Streamlit dashboard lets a user switch between the two feature modes, enter telemetry manually, upload a CSV, view metrics, and inspect explanations.

## Why this is useful

A battery monitoring system can provide an estimate of remaining cycles instead of waiting for a battery to fail. That estimate can support maintenance scheduling, replacement planning, and comparison of battery health.

## Key results

Cycle-aware Random Forest:

- Grouped-CV MAE: 5.80 cycles
- Grouped-CV RMSE: 7.94 cycles
- HNEI_t holdout MAE: 3.24 cycles
- HNEI_t holdout RMSE: 4.07 cycles

Telemetry-only Random Forest:

- Grouped-CV MAE: 30.94 cycles
- Grouped-CV RMSE: 46.96 cycles
- HNEI_t holdout MAE: 73.90 cycles
- HNEI_t holdout RMSE: 92.69 cycles

The difference is expected. The cycle-aware model uses a feature directly related to the way RUL was calculated. The telemetry-only model tests the harder question: how much remaining-life information is present in the measured electrical behavior alone?

## Important limitation

The dataset has only 14 batteries and comes from one source and test setup. The high cycle-aware score should not be presented as proof of generalization to every battery chemistry or manufacturer. More batteries and an external test set are needed for a production claim.

## Commands for a demonstration

```bash
python3 src/preprocess.py
PYTHONPATH=src python3 src/train.py
PYTHONPATH=src python3 src/explain.py --mode both
streamlit run app.py
```
