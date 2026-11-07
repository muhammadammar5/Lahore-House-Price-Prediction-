import numpy as np
import pandas as pd

from src.config import LAHORE_CENTER
from src.features import (ENGINEERED_NUMERIC, FrequencyEncoder, MODEL_INPUT_COLUMNS,
                          N_ENGINEERED, OOFTargetEncoder, add_features, build_preprocessor)
from src.data import clean
from tests.synthetic import make_raw


def row(**kw):
    base = dict(location="DHA Defence", area_sqft=2250.0, bedrooms=4, baths=4,
                latitude=31.47, longitude=74.40, listing_year=2019)
    base.update(kw)
    return pd.DataFrame([base])


def test_engineered_values():
    out = add_features(row(area_sqft=2250.0, bedrooms=3, baths=2)).iloc[0]
    assert np.isclose(out["area_marla"], 10) and np.isclose(out["area_per_bed"], 750)
    assert np.isclose(out["baths_per_bed"], 2 / 3) and out["total_rooms"] == 5 and out["bed_bath_gap"] == 1
    assert out["is_dha"] == 1 and out["is_premium_area"] == 1 and out["is_bahria"] == 0
    assert out["is_kanal_plus"] == 0 and add_features(row(area_sqft=4500.0)).iloc[0]["is_kanal_plus"] == 1


def test_distance_zero_at_city_center_and_invalid_coords_become_nan():
    at_center = add_features(row(latitude=LAHORE_CENTER[0], longitude=LAHORE_CENTER[1])).iloc[0]
    assert at_center["dist_center_km"] < 0.01 and at_center["has_coords"] == 1
    bad = add_features(row(latitude=0.0, longitude=0.0)).iloc[0]   # (0,0) is a common placeholder
    assert np.isnan(bad["dist_center_km"]) and bad["has_coords"] == 0


def test_feature_count_matches_documentation():
    assert N_ENGINEERED == len(ENGINEERED_NUMERIC) + 3 and N_ENGINEERED >= 15


def make_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    locs = rng.choice(list("ABCDE"), n)
    X = pd.DataFrame({"location": locs, "log_area": rng.normal(7, 0.3, n)})
    y = pd.Series(np.where(locs == "A", 17.0, 15.0) + rng.normal(0, 0.1, n))
    return X, y


def test_target_encoder_out_of_fold_does_not_leak_own_target():
    X, y = make_frame()
    enc = OOFTargetEncoder("log_price", n_splits=5, random_state=7)
    before = enc.fit_transform(X, y)[:, 0]
    y_changed = y.copy()
    y_changed.iloc[0] += 50.0                       # wildly change ONE row's target
    after = OOFTargetEncoder("log_price", n_splits=5, random_state=7).fit_transform(X, y_changed)[:, 0]
    assert np.isclose(before[0], after[0])          # that row's own encoding is unaffected


def test_target_encoder_learns_group_means_and_handles_unseen():
    X, y = make_frame()
    enc = OOFTargetEncoder("log_price").fit(X, y)
    out = enc.transform(pd.DataFrame({"location": ["A", "B", "NEVER_SEEN"], "log_area": [7, 7, 7]}))[:, 0]
    assert out[0] > out[1] + 1.0                     # A is much pricier than B
    assert np.isclose(out[2], enc.prior_)            # unseen location falls back to prior


def test_frequency_encoder():
    X, _ = make_frame()
    enc = FrequencyEncoder().fit(X)
    assert enc.transform(pd.DataFrame({"location": ["A", "ZZZ"]}))[1, 0] == 0.0


def test_full_preprocessor_runs_and_names_match():
    df, _ = clean(make_raw(400, seed=5))
    X = add_features(df)[MODEL_INPUT_COLUMNS]
    y = np.log1p(df["price"].to_numpy())
    prep = build_preprocessor()
    Z = prep.fit_transform(X, y)
    names = prep.get_feature_names_out()
    assert Z.shape == (len(X), len(names)) and not np.isnan(Z).any()
    assert {"location_freq", "loc_te_log_price", "loc_te_log_ppsf"} <= set(names)
