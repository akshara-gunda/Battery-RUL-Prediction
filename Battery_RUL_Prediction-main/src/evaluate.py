from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    errors = np.asarray(actual) - np.asarray(predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": float(r2_score(actual, predicted)),
    }
