"""Feature importance and error analysis for the best model.

    python -m src.analyze

Outputs: results/*.csv, results/analysis_summary.json and figures/*.png
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED
from .data import load_split
from .features import MODEL_INPUT_COLUMNS, add_features
from .metrics import rmsle

# Columns that describe the same underlying idea are shuffled together
GROUPS = {
    "Size (area)": ["area_sqft", "log_area", "area_marla", "area_per_bed", "is_kanal_plus"],
    "Rooms (beds/baths)": ["bedrooms", "baths", "baths_per_bed", "total_rooms", "bed_bath_gap"],
    "Location (area name)": ["location"],
    "Location (coordinates)": ["latitude", "longitude", "dist_center_km", "dist_liberty_km", "has_coords"],
    "Neighbourhood flags": ["is_dha", "is_bahria", "is_premium_area"],
    "Listing year": ["listing_year"],
}


def grouped_importance(model, X, y, n_repeats=5, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rmsle(y, model.predict(X))
    rows = []
    for group, cols in GROUPS.items():
        cols = [c for c in cols if c in X.columns]
        increases = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            perm = rng.permutation(len(X))
            shuffled[cols] = X[cols].iloc[perm].to_numpy()
            increases.append(rmsle(y, model.predict(shuffled)) - base)
        rows.append({"group": group, "rmsle_increase": float(np.mean(increases)),
                     "std": float(np.std(increases))})
    return pd.DataFrame(rows).sort_values("rmsle_increase", ascending=False).reset_index(drop=True)


def segment_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    grp = df.groupby(by, observed=True)
    return grp.agg(n=("price", "size"),
                   median_ape_pct=("ape", lambda s: round(float(np.median(s)) * 100, 2)),
                   bias_pct=("signed_err", lambda s: round(float(np.median(s)) * 100, 2)),
                   mae_pkr=("abs_err", lambda s: round(float(np.mean(s)), 0))).reset_index()


def barh(df, label, value, title, path, xlabel, top=15):
    top_df = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(top_df) + 1.5))
    ax.barh(top_df[label], top_df[value], color="#2b7a78")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    card = json.loads((MODELS_DIR / "model_card.json").read_text(encoding="utf-8"))

    test = add_features(load_split("test"))
    X, y = test[MODEL_INPUT_COLUMNS], test["price"].to_numpy()
    pred = np.clip(model.predict(X), 1, None)

    # ------------------------------------------------------ feature importance
    group_df = grouped_importance(model, X, y)
    group_df.to_csv(RESULTS_DIR / "importance_by_group.csv", index=False)
    barh(group_df, "group", "rmsle_increase", f"What drives price? ({card['display_name']})",
         FIGURES_DIR / "importance_by_group.png", "Increase in RMSLE when the group is shuffled")

    perm = permutation_importance(model, X, y, scoring="neg_root_mean_squared_log_error", n_repeats=5,
                                  random_state=SEED, n_jobs=1)
    feat_df = pd.DataFrame({"feature": X.columns, "rmsle_increase": perm.importances_mean,
                            "std": perm.importances_std}).sort_values("rmsle_increase", ascending=False)
    feat_df.to_csv(RESULTS_DIR / "importance_by_feature.csv", index=False)
    barh(feat_df, "feature", "rmsle_increase", "Permutation importance by input column",
         FIGURES_DIR / "importance_by_feature.png", "Increase in RMSLE when the column is shuffled")

    final = model.regressor_.named_steps["model"]
    if hasattr(final, "feature_importances_"):
        names = model.regressor_.named_steps["prep"].get_feature_names_out()
        native = pd.DataFrame({"feature": names, "importance": final.feature_importances_})
        native.sort_values("importance", ascending=False).to_csv(RESULTS_DIR / "importance_native.csv", index=False)

    # ----------------------------------------------------------- error analysis
    df = test[["location", "area_sqft", "bedrooms", "baths", "price"]].copy()
    df["pred"] = pred
    df["abs_err"] = (df["pred"] - df["price"]).abs()
    df["signed_err"] = (df["pred"] - df["price"]) / df["price"]
    df["ape"] = df["signed_err"].abs()
    df["price_band"] = pd.qcut(df["price"], 4, labels=["Q1 cheapest", "Q2", "Q3", "Q4 most expensive"])
    df["area_band"] = pd.cut(df["area_sqft"] / 225, [0, 5, 7, 10, 20, 40, np.inf],
                             labels=["<=5 marla", "5-7", "7-10", "10-20 (1 kanal)", "20-40", "40+"])

    by_price = segment_table(df, "price_band")
    by_area = segment_table(df, "area_band")
    counts = df["location"].value_counts()
    top_locs = counts[counts >= 20].head(12).index
    by_loc = segment_table(df[df["location"].isin(top_locs)], "location").sort_values("n", ascending=False)
    by_price.to_csv(RESULTS_DIR / "error_by_price_band.csv", index=False)
    by_area.to_csv(RESULTS_DIR / "error_by_area.csv", index=False)
    by_loc.to_csv(RESULTS_DIR / "error_by_location.csv", index=False)
    df.sort_values("ape", ascending=False).head(25)[
        ["location", "area_sqft", "bedrooms", "baths", "price", "pred", "ape"]
    ].to_csv(RESULTS_DIR / "worst_predictions.csv", index=False)

    # --------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["price"], df["pred"], s=6, alpha=0.35, color="#2b7a78")
    lim = [df["price"].min() * 0.8, df["price"].max() * 1.2]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual price (PKR)"); ax.set_ylabel("Predicted price (PKR)")
    ax.set_title("Predicted vs actual (test set)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "pred_vs_actual.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log(df["pred"] / df["price"]), bins=60, color="#2b7a78")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("log(predicted / actual)   (0 = perfect)"); ax.set_ylabel("Listings")
    ax.set_title("Error distribution (test set)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "error_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(by_price["price_band"].astype(str), by_price["median_ape_pct"], color="#2b7a78")
    ax.set_ylabel("Median absolute % error"); ax.set_title("Error by price band")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "error_by_price_band.png", dpi=150); plt.close(fig)

    summary = {
        "best_model": card["display_name"],
        "top_price_drivers": group_df.head(3)["group"].tolist(),
        "top_features": feat_df.head(5)["feature"].tolist(),
        "median_ape_pct_overall": round(float(df["ape"].median()) * 100, 2),
        "share_within_10pct": round(float((df["ape"] <= 0.10).mean()) * 100, 1),
        "share_within_20pct": round(float((df["ape"] <= 0.20).mean()) * 100, 1),
        "median_ape_pct_by_price_band": dict(zip(by_price["price_band"].astype(str), by_price["median_ape_pct"])),
    }
    (RESULTS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nImportance by group:\n", group_df.to_string(index=False))
    print("\nError by price band:\n", by_price.to_string(index=False))


if __name__ == "__main__":
    main()
