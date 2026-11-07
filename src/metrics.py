"""Regression metrics. Prices span PKR 1 million to 3 billion, so we report both
PKR-scale errors (dominated by luxury homes) and log-scale errors (relative accuracy)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score


def _clip(pred):
    return np.clip(np.asarray(pred, dtype=float), 1.0, None)


def rmse(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - _clip(pred)) ** 2)))


def mae(y, pred) -> float:
    return float(mean_absolute_error(y, _clip(pred)))


def rmsle(y, pred) -> float:
    return float(np.sqrt(np.mean((np.log1p(np.asarray(y, float)) - np.log1p(_clip(pred))) ** 2)))


def r2_log(y, pred) -> float:
    return float(r2_score(np.log1p(np.asarray(y, float)), np.log1p(_clip(pred))))


def median_ape(y, pred) -> float:
    y = np.asarray(y, float)
    return float(np.median(np.abs(y - _clip(pred)) / y))


def regression_report(y, pred) -> dict:
    pred = _clip(pred)
    return {
        "rmse_pkr": round(rmse(y, pred), 0),
        "mae_pkr": round(mae(y, pred), 0),
        "r2": round(float(r2_score(y, pred)), 4),
        "r2_log": round(r2_log(y, pred), 4),
        "rmsle": round(rmsle(y, pred), 4),
        "median_ape_pct": round(median_ape(y, pred) * 100, 2),
    }


SCORERS = {
    "rmse": make_scorer(rmse, greater_is_better=False),
    "mae": make_scorer(mae, greater_is_better=False),
    "r2": make_scorer(r2_score),
    "r2_log": make_scorer(r2_log),
    "rmsle": make_scorer(rmsle, greater_is_better=False),
    "median_ape": make_scorer(median_ape, greater_is_better=False),
}
RMSLE_SCORER = SCORERS["rmsle"]
