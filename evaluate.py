"""Rolling-origin errors of the persistence forecast, and safety margins (training rows only)."""
import numpy as np


def rolling_errors(series, h=24, first=30):
    y = np.asarray(series, dtype=float)
    first = max(1, min(first, len(y) - h))
    errors, steps = [], []
    for start in range(first, len(y) - h + 1):
        last = y[start - 1]
        for i in range(1, h + 1):
            errors.append(y[start + i - 1] - last)        # actual - forecast
            steps.append(i)
    return np.array(errors), np.array(steps)


def summary(errors):
    return {"MAE": float(np.abs(errors).mean()),
            "RMSE": float(np.sqrt((errors ** 2).mean())),
            "margin_90": float(np.quantile(errors, 0.90)),
            "margin_95": float(np.quantile(errors, 0.95))}
