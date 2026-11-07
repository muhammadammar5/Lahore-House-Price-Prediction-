"""Load, clean and split Zameen.com property listings for Lahore.

    python -m src.data                                   # uses data/raw/zameen.csv
    python -m src.data --raw-path path/to/file.csv --numeric-area-unit sqft

The loader adapts to the common public Zameen datasets (Kaggle / opendata.com.pk),
which use slightly different column names. Area may be text ("5 Marla") or a
number of square feet. Prices may be numbers or text such as "PKR 1.2 Crore".

All filters are fixed domain rules (see config.BOUNDS), applied BEFORE the
train/test split, and every step's row count is saved to stats.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (BOUNDS, KANAL_SQFT, MARLA_SQFT, PROCESSED_DIR, RAW_PATH, SEED,
                     TEST_SIZE)

# standard name -> accepted raw column names (lower-case)
ALIASES: Dict[str, list] = {
    "property_id": ["property_id", "id", "listing_id"],
    "property_type": ["property_type", "type"],
    "price": ["price"],
    "location": ["location", "area_name", "locality"],
    "city": ["city"],
    "purpose": ["purpose", "listing_type"],
    "bedrooms": ["bedrooms", "beds", "bedroom"],
    "baths": ["baths", "bathrooms", "bath"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng", "long"],
    "date_added": ["date_added", "date", "added"],
}
AREA_TEXT_COLS = ["area", "area_size", "size"]
AREA_SQFT_COLS = ["total_area", "area_sqft", "areasquarefeet", "area_in_sqft", "area_sq_ft"]

_UNIT_RE = re.compile(
    r"(?P<num>\d[\d,]*\.?\d*)\s*(?P<unit>marla|kanal|sq\.?\s*ft|sqft|square\s*feet|"
    r"sq\.?\s*yd|square\s*yards?|sq\.?\s*m|square\s*met(?:er|re)s?)",
    re.IGNORECASE,
)
_UNIT_TO_SQFT = {"marla": MARLA_SQFT, "kanal": KANAL_SQFT, "sqft": 1.0, "sqyd": 9.0, "sqm": 10.7639}
_PRICE_RE = re.compile(r"(?P<num>\d[\d,]*\.?\d*)\s*(?P<unit>arab|crore|lakh|lac|thousand)?", re.IGNORECASE)
_PRICE_MULT = {"arab": 1e9, "crore": 1e7, "lakh": 1e5, "lac": 1e5, "thousand": 1e3, None: 1.0}


def _unit_key(unit: str) -> str:
    u = re.sub(r"[\s.]", "", unit.lower())
    if u.startswith("marla"):
        return "marla"
    if u.startswith("kanal"):
        return "kanal"
    if u in ("sqft", "squarefeet"):
        return "sqft"
    if u.startswith("sqy") or u.startswith("squareyard"):
        return "sqyd"
    return "sqm"


def parse_area(series: pd.Series, numeric_unit: str = "sqft") -> pd.Series:
    """Convert an area column (text with units, or plain numbers) to square feet."""
    factor = {"sqft": 1.0, "marla": MARLA_SQFT, "kanal": KANAL_SQFT}[numeric_unit]
    as_num = pd.to_numeric(series, errors="coerce")
    out = as_num * factor
    text_mask = as_num.isna() & series.notna()
    if text_mask.any():
        extracted = series[text_mask].astype(str).str.extract(_UNIT_RE)
        nums = pd.to_numeric(extracted["num"].str.replace(",", "", regex=False), errors="coerce")
        mult = extracted["unit"].map(lambda u: _UNIT_TO_SQFT[_unit_key(u)] if isinstance(u, str) else np.nan)
        out.loc[text_mask] = nums * mult
    return out.astype(float)


def parse_price(series: pd.Series) -> pd.Series:
    """Convert prices to PKR. Handles numbers and text like 'PKR 1.2 Crore'."""
    as_num = pd.to_numeric(series, errors="coerce")
    out = as_num.copy()
    text_mask = as_num.isna() & series.notna()
    if text_mask.any():
        extracted = series[text_mask].astype(str).str.extract(_PRICE_RE)
        nums = pd.to_numeric(extracted["num"].str.replace(",", "", regex=False), errors="coerce")
        mult = extracted["unit"].map(lambda u: _PRICE_MULT[u.lower()] if isinstance(u, str) else 1.0)
        out.loc[text_mask] = nums * mult
    return out.astype(float)


def _find(df_cols, candidates) -> Optional[str]:
    for cand in candidates:
        if cand in df_cols:
            return cand
    return None


def standardize(raw: pd.DataFrame, numeric_area_unit: str = "sqft") -> pd.DataFrame:
    """Map a raw Zameen CSV to the standard columns used by this project."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = set(df.columns)

    out = pd.DataFrame(index=df.index)
    for std, options in ALIASES.items():
        src = _find(cols, options)
        out[std] = df[src] if src else np.nan

    # area: prefer text-with-units, fall back to a numeric square-feet column
    area = pd.Series(np.nan, index=df.index)
    text_col = _find(cols, AREA_TEXT_COLS)
    if text_col:
        area = parse_area(df[text_col], numeric_area_unit)
    sqft_col = _find(cols, AREA_SQFT_COLS)
    if sqft_col:
        area = area.fillna(parse_area(df[sqft_col], "sqft"))
    out["area_sqft"] = area

    out["price"] = parse_price(out["price"])
    for col in ("bedrooms", "baths", "latitude", "longitude"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date_added"] = pd.to_datetime(out["date_added"], errors="coerce", format="mixed")
    out["listing_year"] = out["date_added"].dt.year.astype(float)

    missing = [c for c in ("price", "area_sqft", "bedrooms", "location") if out[c].isna().all()]
    if missing:
        raise ValueError(
            f"Could not find usable columns for {missing}. Columns in your file: {sorted(cols)}"
        )
    return out


def clean(raw: pd.DataFrame, numeric_area_unit: str = "sqft",
          property_types=("house",), city: str = "lahore"):
    """Standardise, filter to Lahore houses for sale, apply rule-based bounds, de-duplicate."""
    stats: Dict[str, int] = {"raw_rows": int(len(raw))}
    df = standardize(raw, numeric_area_unit)

    if df["city"].notna().any():
        df = df[df["city"].astype(str).str.strip().str.lower().str.contains(city)]
    stats[f"city_{city}"] = int(len(df))

    if df["purpose"].notna().any():
        df = df[df["purpose"].astype(str).str.lower().str.contains("sale")]
    stats["for_sale"] = int(len(df))

    if df["property_type"].notna().any() and property_types:
        wanted = {p.lower() for p in property_types}
        df = df[df["property_type"].astype(str).str.strip().str.lower().isin(wanted)]
    stats["property_type_match"] = int(len(df))

    df = df.dropna(subset=["price", "area_sqft", "bedrooms", "location"])
    df["location"] = df["location"].astype(str).str.strip()
    df = df[df["location"].str.len() > 0]
    stats["after_dropping_missing"] = int(len(df))

    df["price_per_sqft"] = df["price"] / df["area_sqft"]
    keep = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in BOUNDS.items():
        values = df[col]
        if col == "baths":                      # baths may be missing; imputed later
            keep &= (values.isna() | values.between(lo, hi)).to_numpy()
        else:
            keep &= values.between(lo, hi).to_numpy()
    df = df[keep]
    stats["after_rule_based_bounds"] = int(len(df))

    if df["property_id"].notna().any():
        df = df.drop_duplicates(subset="property_id", keep="first")
    else:
        df = df.drop_duplicates(subset=["location", "price", "area_sqft", "bedrooms", "baths"])
    stats["after_deduplication"] = int(len(df))
    stats["final_rows"] = stats["after_deduplication"]

    keep_cols = ["property_id", "location", "price", "area_sqft", "bedrooms", "baths",
                 "latitude", "longitude", "date_added", "listing_year"]
    return df[keep_cols].reset_index(drop=True), stats


def split(df: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE):
    """Random split, stratified on price quartile so both sets cover cheap and luxury homes."""
    strata = pd.qcut(df["price"], q=4, labels=False, duplicates="drop")
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare(raw_path: Path = RAW_PATH, out_dir: Path = PROCESSED_DIR,
            numeric_area_unit: str = "sqft", property_types=("house",),
            seed: int = SEED) -> Dict[str, object]:
    raw = pd.read_csv(raw_path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df, stats = clean(raw, numeric_area_unit, property_types)
    if len(df) < 200:
        raise ValueError(f"Only {len(df)} usable rows after cleaning; check your file and filters.")
    train, test = split(df, seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    stats.update(
        seed=seed,
        splits={"train": int(len(train)), "test": int(len(test))},
        price_pkr_quantiles={str(q): float(df["price"].quantile(q)) for q in (0.05, 0.5, 0.95)},
        top_locations=df["location"].value_counts().head(10).to_dict(),
        year_range=[float(df["listing_year"].min()), float(df["listing_year"].max())]
        if df["listing_year"].notna().any() else None,
    )
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_split(name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    path = Path(processed_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Lahore house-price data")
    parser.add_argument("--raw-path", default=str(RAW_PATH))
    parser.add_argument("--numeric-area-unit", choices=["sqft", "marla", "kanal"], default="sqft",
                        help="Unit to assume when the area column holds plain numbers")
    parser.add_argument("--property-types", nargs="+", default=["House"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    stats = prepare(Path(args.raw_path), PROCESSED_DIR, args.numeric_area_unit,
                    tuple(args.property_types), args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
