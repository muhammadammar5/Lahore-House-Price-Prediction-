"""Compare 5 models, tune them with 5-fold CV, evaluate once on the held-out test set.

    python -m src.train                  # full run (about 20-40 min on a laptop CPU)
    python -m src.train --fast           # quick smoke test on a small sample

Stages (every model and every tuning trial is logged to MLflow):
  1. Compare 5 models with default settings using 5-fold cross-validation (train data only)
  2. Tune each model with 5-fold randomized search, optimising RMSLE (relative error)
  3. Evaluate every tuned model ONCE on the untouched test set
The "best" model is chosen by cross-validation score, never by test score.
View runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate

from .config import (CV_FOLDS, MLFLOW_EXPERIMENT, MLFLOW_URI, MODELS_DIR, RESULTS_DIR, SEED)
from .data import load_split
from .features import MODEL_INPUT_COLUMNS, N_ENGINEERED, add_features
from .metrics import RMSLE_SCORER, SCORERS, regression_report
from .models import DEFAULT_N_ITER, DISPLAY, MODEL_NAMES, PREFIX, build_model, param_space

NEGATED = ("rmse", "mae", "rmsle", "median_ape")  # scorers where sklearn returns -value


def log_child(name: str, params: dict, metrics: dict, tags: dict | None = None) -> None:
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)})
        for key, value in (tags or {}).items():
            mlflow.set_tag(key, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    parser.add_argument("--tune", nargs="+", default=["ridge", "random_forest", "gradient_boosting", "xgboost"],
                        choices=MODEL_NAMES, help="Models to tune (SVR is slow to tune, so skipped by default)")
    parser.add_argument("--n-iter", type=int, default=None, help="Override random-search trials per model")
    parser.add_argument("--cv", type=int, default=CV_FOLDS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=None, help="Use only N training rows (debugging)")
    parser.add_argument("--fast", action="store_true", help="Tiny run: 3000 rows, 3 folds, 2 trials per model")
    args = parser.parse_args()
    if args.fast:
        args.sample, args.cv, args.n_iter = 3000, 3, 2

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    train = add_features(load_split("train"))
    test = add_features(load_split("test"))
    if args.sample and len(train) > args.sample:
        train = train.sample(args.sample, random_state=SEED).reset_index(drop=True)
    X_train, y_train = train[MODEL_INPUT_COLUMNS], train["price"].to_numpy()
    X_test, y_test = test[MODEL_INPUT_COLUMNS], test["price"].to_numpy()
    print(f"Train rows: {len(train)} | test rows: {len(test)} | engineered features: {N_ENGINEERED}")

    cv = KFold(args.cv, shuffle=True, random_state=SEED)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ------------------------------------------------------------ stage 1: compare
    cv_rows = []
    with mlflow.start_run(run_name="1_model_comparison"):
        mlflow.log_params({"cv_folds": args.cv, "n_train": len(train), "stage": "compare"})
        for name in args.models:
            start = time.perf_counter()
            res = cross_validate(build_model(name, SEED, args.n_jobs), X_train, y_train, cv=cv,
                                 scoring=SCORERS, n_jobs=1)
            row = {"model": name}
            for metric in SCORERS:
                value = float(np.mean(res[f"test_{metric}"]))
                row[metric] = -value if metric in NEGATED else value
                row[f"{metric}_std"] = float(np.std(res[f"test_{metric}"]))
            row["seconds"] = round(time.perf_counter() - start, 1)
            cv_rows.append(row)
            log_child(f"compare_{name}", {"model": name, "stage": "compare"},
                      {k: v for k, v in row.items() if k not in ("model", "seconds") and not k.endswith("_std")})
            print(f"[compare] {DISPLAY[name]:<30} CV RMSLE={row['rmsle']:.4f}  R2(log)={row['r2_log']:.4f}  "
                  f"R2={row['r2']:.4f}  ({row['seconds']}s)")
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(RESULTS_DIR / "cv_comparison.csv", index=False)

    # -------------------------------------------------------------- stage 2: tune
    fitted, tuning_rows, best_params = {}, [], {}
    with mlflow.start_run(run_name="2_hyperparameter_tuning"):
        for name in args.models:
            if name not in args.tune:
                fitted[name] = build_model(name, SEED, args.n_jobs).fit(X_train, y_train)
                continue
            n_iter = args.n_iter or DEFAULT_N_ITER[name]
            start = time.perf_counter()
            search = RandomizedSearchCV(build_model(name, SEED, args.n_jobs), param_space(name), n_iter=n_iter,
                                        cv=cv, scoring=RMSLE_SCORER, random_state=SEED, n_jobs=1, refit=True)
            search.fit(X_train, y_train)
            results = search.cv_results_
            for i in range(len(results["params"])):
                trial = {k.replace(PREFIX, ""): v for k, v in results["params"][i].items()}
                log_child(f"tune_{name}_{i:02d}", {"model": name, "stage": "tune", **trial},
                          {"cv_rmsle": -results["mean_test_score"][i],
                           "cv_rmsle_std": results["std_test_score"][i],
                           "fit_seconds": results["mean_fit_time"][i]})
            fitted[name] = search.best_estimator_
            best_params[name] = {k.replace(PREFIX, ""): (v.item() if hasattr(v, "item") else v)
                                 for k, v in search.best_params_.items()}
            tuning_rows.append({"model": name, "trials": n_iter, "best_cv_rmsle": round(-search.best_score_, 4),
                                "seconds": round(time.perf_counter() - start, 1)})
            print(f"[tune]    {DISPLAY[name]:<30} best CV RMSLE={-search.best_score_:.4f} ({n_iter} trials x {args.cv} folds)")
    pd.DataFrame(tuning_rows).to_csv(RESULTS_DIR / "tuning_summary.csv", index=False)
    (RESULTS_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------ stage 3: final test eval
    tuned_cv = {t["model"]: t["best_cv_rmsle"] for t in tuning_rows}
    final_cv = {r["model"]: tuned_cv.get(r["model"], r["rmsle"]) for r in cv_rows}

    test_rows, preds = [], {}
    dummy = TransformedTargetRegressor(regressor=DummyRegressor(strategy="median"), func=np.log1p,
                                       inverse_func=np.expm1, check_inverse=False).fit(X_train, y_train)
    candidates = {"dummy_median": dummy, **fitted}
    with mlflow.start_run(run_name="3_final_test_evaluation"):
        for name, model in candidates.items():
            pred = model.predict(X_test)
            preds[name] = pred
            report = regression_report(y_test, pred)
            test_rows.append({"model": name, "cv_rmsle": final_cv.get(name), **report})
            log_child(f"test_{name}", {"model": name, "stage": "test"}, {f"test_{k}": v for k, v in report.items()})

    test_df = pd.DataFrame(test_rows)
    if "ridge" in preds:
        base = test_df.loc[test_df["model"] == "ridge"].iloc[0]
        test_df["rmse_reduction_vs_ridge_pct"] = ((1 - test_df["rmse_pkr"] / base["rmse_pkr"]) * 100).round(1)
        test_df["rmsle_reduction_vs_ridge_pct"] = ((1 - test_df["rmsle"] / base["rmsle"]) * 100).round(1)
    test_df.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    (RESULTS_DIR / "test_metrics.json").write_text(test_df.to_json(orient="records", indent=2), encoding="utf-8")

    # ------------------------------------------- pick best by CV (NOT by test score)
    best_name = min(final_cv, key=final_cv.get)
    best_model = fitted[best_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    card = {
        "best_model": best_name, "display_name": DISPLAY[best_name],
        "selected_by": "lowest cross-validated RMSLE on the training split",
        "cv_rmsle": final_cv[best_name],
        "test_metrics": test_df.loc[test_df["model"] == best_name].iloc[0].to_dict(),
        "input_columns": MODEL_INPUT_COLUMNS,
        "reference_year": float(train["listing_year"].max()) if train["listing_year"].notna().any() else None,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")

    out = test[["location", "area_sqft", "bedrooms", "baths", "price"]].copy()
    out["predicted_price"] = np.clip(preds[best_name], 1, None).round(0)
    out.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    n_runs = len(mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT]))
    print("\nTEST SET RESULTS (price in PKR)")
    print(test_df.to_string(index=False))
    print(f"\nBest model by cross-validation: {DISPLAY[best_name]}")
    print(f"MLflow runs logged: {n_runs}  (view with: mlflow ui --backend-store-uri {MLFLOW_URI})")


if __name__ == "__main__":
    main()
