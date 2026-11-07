"""Exploratory plots from the TRAINING split only (the test set stays unseen).

    python -m src.eda
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES_DIR, RESULTS_DIR
from .data import load_split


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_split("train")
    df["ppsf"] = df["price"] / df["area_sqft"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(np.log10(df["price"]), bins=60, color="#2b7a78")
    ticks = [6, 7, 8, 9]
    ax.set_xticks(ticks, ["10 lakh", "1 crore", "10 crore", "100 crore"])
    ax.set_xlabel("Asking price (log scale)"); ax.set_ylabel("Listings")
    ax.set_title("Price distribution is heavily right-skewed")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_price_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["area_sqft"] / 225, df["price"], s=4, alpha=0.25, color="#2b7a78")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Area (marla, log)"); ax.set_ylabel("Price (PKR, log)")
    ax.set_title("Price vs area")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_price_vs_area.png", dpi=150); plt.close(fig)

    counts = df["location"].value_counts()
    top = counts[counts >= 30].head(15).index
    med = df[df["location"].isin(top)].groupby("location")["ppsf"].median().sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(med.index, med.values, color="#2b7a78")
    ax.set_xlabel("Median price per sq ft (PKR)")
    ax.set_title("Price per sq ft in the 15 most-listed areas")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "eda_price_per_sqft_by_area.png", dpi=150); plt.close(fig)

    summary = {
        "rows": int(len(df)),
        "median_price_pkr": float(df["price"].median()),
        "median_price_per_sqft": float(df["ppsf"].median()),
        "n_locations": int(df["location"].nunique()),
        "missing_share": {c: round(float(df[c].isna().mean()), 3)
                          for c in ("baths", "latitude", "longitude", "listing_year")},
    }
    (RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
