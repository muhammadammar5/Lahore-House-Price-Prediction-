"""Estimate the asking price of one house.

    python -m src.predict --location "DHA Defence" --area 10 --unit marla --bedrooms 4 --baths 4
    python -m src.predict --location "Johar Town" --area 1 --unit kanal --bedrooms 5 --baths 5 --lat 31.47 --lon 74.28
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import KANAL_SQFT, MARLA_SQFT, MODELS_DIR
from .features import MODEL_INPUT_COLUMNS, add_features

UNITS = {"marla": MARLA_SQFT, "kanal": KANAL_SQFT, "sqft": 1.0}


def format_pkr(value: float) -> str:
    if value >= 1e7:
        return f"PKR {value / 1e7:.2f} crore"
    return f"PKR {value / 1e5:.1f} lakh"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", required=True, help='Area name as in the data, e.g. "DHA Defence"')
    parser.add_argument("--area", type=float, required=True)
    parser.add_argument("--unit", choices=list(UNITS), default="marla")
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--baths", type=int, required=True)
    parser.add_argument("--lat", type=float, default=np.nan)
    parser.add_argument("--lon", type=float, default=np.nan)
    parser.add_argument("--year", type=float, default=None, help="Listing year (default: latest in training data)")
    args = parser.parse_args()

    model = joblib.load(MODELS_DIR / "best_model.joblib")
    card = json.loads((MODELS_DIR / "model_card.json").read_text(encoding="utf-8"))

    row = pd.DataFrame([{
        "location": args.location, "area_sqft": args.area * UNITS[args.unit], "bedrooms": args.bedrooms,
        "baths": args.baths, "latitude": args.lat, "longitude": args.lon,
        "listing_year": args.year if args.year is not None else card.get("reference_year", np.nan),
    }])
    price = float(np.clip(model.predict(add_features(row)[MODEL_INPUT_COLUMNS])[0], 1, None))
    print(f"Estimated asking price: {format_pkr(price)}  (PKR {price:,.0f})")
    print(f"Model: {card['display_name']} | typical error: about "
          f"{card['test_metrics']['median_ape_pct']}% (median absolute % error on the test set)")


if __name__ == "__main__":
    main()
