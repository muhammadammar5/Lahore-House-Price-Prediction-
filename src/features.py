"""Feature engineering.

Two kinds of features:
  * add_features(): stateless, row-wise features (safe to compute before splitting)
  * encoders inside the sklearn pipeline: statistics learned from the TRAINING fold
    only (location frequency, smoothed target encodings). Target encodings are
    computed out-of-fold on the training data so a row never sees its own price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import LAHORE_BBOX, LAHORE_CENTER, LIBERTY_MARKET, MARLA_SQFT, SEED

# Raw columns that go into the model unchanged
RAW_NUMERIC = ["area_sqft", "bedrooms", "baths", "latitude", "longitude", "listing_year"]
# Stateless engineered columns created by add_features()
ENGINEERED_NUMERIC = [
    "log_area", "area_marla", "baths_per_bed", "area_per_bed", "total_rooms",
    "bed_bath_gap", "is_kanal_plus", "dist_center_km", "dist_liberty_km",
    "has_coords", "is_dha", "is_bahria", "is_premium_area",
]
# Engineered inside the pipeline (fit on training data only)
STATEFUL_FEATURES = ["location_freq", "loc_te_log_price", "loc_te_log_ppsf"]

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
N_ENGINEERED = len(ENGINEERED_NUMERIC) + len(STATEFUL_FEATURES)

PREMIUM_KEYWORDS = ("dha", "defence", "gulberg", "model town", "cantt", "askari")


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stateless engineered columns. Returns a new DataFrame."""
    out = df.copy()
    for col in RAW_NUMERIC:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    lat_min, lat_max, lon_min, lon_max = LAHORE_BBOX
    valid = (out["latitude"].between(lat_min, lat_max) & out["longitude"].between(lon_min, lon_max))
    out["latitude"] = out["latitude"].where(valid)
    out["longitude"] = out["longitude"].where(valid)

    beds = out["bedrooms"].clip(lower=1)
    out["log_area"] = np.log(out["area_sqft"])
    out["area_marla"] = out["area_sqft"] / MARLA_SQFT
    out["baths_per_bed"] = out["baths"] / beds
    out["area_per_bed"] = out["area_sqft"] / beds
    out["total_rooms"] = out["bedrooms"] + out["baths"]
    out["bed_bath_gap"] = out["bedrooms"] - out["baths"]
    out["is_kanal_plus"] = (out["area_marla"] >= 20).astype(float)
    out["dist_center_km"] = haversine_km(out["latitude"], out["longitude"], *LAHORE_CENTER)
    out["dist_liberty_km"] = haversine_km(out["latitude"], out["longitude"], *LIBERTY_MARKET)
    out["has_coords"] = valid.astype(float)

    loc = out["location"].astype(str).str.lower()
    out["is_dha"] = loc.str.contains(r"\bdha\b|defence", regex=True).astype(float)
    out["is_bahria"] = loc.str.contains("bahria").astype(float)
    out["is_premium_area"] = loc.apply(lambda s: float(any(k in s for k in PREMIUM_KEYWORDS)))
    return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """log(1 + number of training listings in the same location)."""

    def fit(self, X, y=None):
        self.counts_ = X["location"].astype(str).value_counts().to_dict()
        return self

    def transform(self, X):
        counts = X["location"].astype(str).map(self.counts_).fillna(0).to_numpy(float)
        return np.log1p(counts).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["location_freq"])


class OOFTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed mean target per location, computed out-of-fold when fitting.

    target="log_price": mean log price in the location
    target="log_ppsf" : mean log price-per-sqft (log price minus log area)

    fit_transform() (used on training data) encodes each row from the OTHER folds,
    so a listing's own price never leaks into its feature. transform() (used on new
    data) uses statistics from the full training data; unseen locations get the prior.
    Expects columns "location" and "log_area".
    """

    def __init__(self, target: str = "log_price", smooth: float = 10.0,
                 n_splits: int = 5, random_state: int = SEED):
        self.target = target
        self.smooth = smooth
        self.n_splits = n_splits
        self.random_state = random_state

    def _target(self, X, y):
        y = np.asarray(y, dtype=float)
        return y if self.target == "log_price" else y - X["log_area"].to_numpy(float)

    def _stats(self, loc, t):
        prior = float(np.mean(t))
        grouped = pd.DataFrame({"loc": loc, "t": t}).groupby("loc")["t"].agg(["sum", "count"])
        enc = (grouped["sum"] + self.smooth * prior) / (grouped["count"] + self.smooth)
        return enc.to_dict(), prior

    def fit(self, X, y):
        t = self._target(X, y)
        self.map_, self.prior_ = self._stats(X["location"].astype(str).to_numpy(), t)
        return self

    def transform(self, X):
        vals = X["location"].astype(str).map(self.map_).fillna(self.prior_).to_numpy(float)
        return vals.reshape(-1, 1)

    def fit_transform(self, X, y=None, **fit_params):
        X = X.reset_index(drop=True)
        t = self._target(X, y)
        loc = X["location"].astype(str).to_numpy()
        out = np.empty(len(X))
        kfold = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kfold.split(X):
            mapping, prior = self._stats(loc[train_idx], t[train_idx])
            out[val_idx] = pd.Series(loc[val_idx]).map(mapping).fillna(prior).to_numpy(float)
        self.fit(X, y)
        return out.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"loc_te_{self.target}"])


def build_preprocessor() -> ColumnTransformer:
    """Imputation + scaling for numeric columns, plus the stateful location encoders."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("freq", Pipeline([("enc", FrequencyEncoder()), ("scale", StandardScaler())]), ["location"]),
            ("te_price", Pipeline([("enc", OOFTargetEncoder("log_price")), ("scale", StandardScaler())]),
             ["location", "log_area"]),
            ("te_ppsf", Pipeline([("enc", OOFTargetEncoder("log_ppsf")), ("scale", StandardScaler())]),
             ["location", "log_area"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


MODEL_INPUT_COLUMNS = NUMERIC_FEATURES + ["location"]
