"""Model zoo and hyper-parameter search spaces.

Every model is wrapped as: log1p(price) target -> preprocessing -> regressor -> expm1,
so predictions come back in PKR while the model learns relative (log) price.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .features import build_preprocessor

MODEL_NAMES = ["ridge", "random_forest", "gradient_boosting", "xgboost", "svr"]
DISPLAY = {
    "ridge": "Ridge regression (baseline)",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting (HistGB)",
    "xgboost": "XGBoost",
    "svr": "Support Vector Regression",
}
# Default number of random-search candidates per model (5-fold CV each)
DEFAULT_N_ITER = {"ridge": 6, "random_forest": 12, "gradient_boosting": 20, "xgboost": 30}


def make_estimator(name: str, seed: int, n_jobs: int):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=n_jobs,
                                     random_state=seed)
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed)
    if name == "xgboost":
        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs,
                            random_state=seed, verbosity=0)
    if name == "svr":
        return SVR(C=10.0, epsilon=0.05, gamma="scale")
    raise ValueError(f"Unknown model: {name}")


def build_model(name: str, seed: int = 42, n_jobs: int = -1) -> TransformedTargetRegressor:
    pipe = Pipeline([("prep", build_preprocessor()), ("model", make_estimator(name, seed, n_jobs))])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1,
                                      check_inverse=False)


PREFIX = "regressor__model__"


def param_space(name: str) -> Dict[str, object]:
    spaces = {
        "ridge": {"alpha": loguniform(1e-2, 1e2)},
        "random_forest": {
            "n_estimators": randint(150, 500), "max_depth": randint(8, 30),
            "min_samples_leaf": randint(1, 8), "max_features": uniform(0.3, 0.7),
        },
        "gradient_boosting": {
            "learning_rate": loguniform(0.02, 0.3), "max_iter": randint(150, 700),
            "max_leaf_nodes": randint(15, 80), "min_samples_leaf": randint(10, 60),
            "l2_regularization": loguniform(1e-3, 10),
        },
        "xgboost": {
            "n_estimators": randint(200, 900), "learning_rate": loguniform(0.02, 0.2),
            "max_depth": randint(3, 10), "min_child_weight": randint(1, 10),
            "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
            "reg_lambda": loguniform(0.1, 20), "reg_alpha": loguniform(1e-3, 5),
        },
    }
    return {PREFIX + k: v for k, v in spaces.get(name, {}).items()}
