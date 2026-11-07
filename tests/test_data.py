import pandas as pd

from src.data import clean, parse_area, parse_price, prepare, split, standardize
from tests.synthetic import make_raw


def test_parse_area_units():
    s = pd.Series(["5 Marla", "1 Kanal", "2,250 sqft", "10 Sq. Yd.", "garbage", None])
    out = parse_area(s).tolist()
    assert out[0] == 5 * 225 and out[1] == 4500 and out[2] == 2250 and out[3] == 90
    assert pd.isna(out[4]) and pd.isna(out[5])


def test_parse_area_plain_numbers_use_chosen_unit():
    assert parse_area(pd.Series([10]), "marla").iloc[0] == 2250
    assert parse_area(pd.Series([10]), "sqft").iloc[0] == 10


def test_parse_price_handles_text_and_numbers():
    out = parse_price(pd.Series([12500000, "PKR 1.2 Crore", "85 Lakh", "2 Arab", "3,500,000"])).tolist()
    assert out == [12500000, 12000000, 8500000, 2e9, 3500000]


def test_standardize_accepts_alternative_column_names():
    raw = pd.DataFrame({"Price": [1e7], "Beds": [3], "Bathrooms": [2], "Locality": ["DHA"],
                        "Area": ["5 Marla"], "City": ["Lahore"], "Lat": [31.5], "Lng": [74.3]})
    out = standardize(raw)
    assert out.loc[0, "bedrooms"] == 3 and out.loc[0, "baths"] == 2 and out.loc[0, "area_sqft"] == 1125


def test_clean_filters_noise_and_duplicates():
    raw = make_raw(800, seed=1)
    df, stats = clean(raw)
    assert (df["price"] >= 1_000_000).all()                 # absurd prices removed
    assert df["property_id"].is_unique                      # duplicates removed
    assert stats["final_rows"] == len(df) == 800            # all noise rows removed, originals kept
    assert stats["city_lahore"] < stats["raw_rows"]


def test_numeric_area_column_fallback():
    raw = make_raw(300, seed=2).drop(columns=["area"])      # only Total_Area (sq ft) remains
    df, _ = clean(raw)
    assert df["area_sqft"].between(450, 45_000).all()


def test_split_is_disjoint_and_sized(tmp_path):
    df, _ = clean(make_raw(600, seed=3))
    train, test = split(df, seed=1)
    assert len(train) + len(test) == len(df)
    assert not set(train["property_id"]) & set(test["property_id"])
    assert abs(len(test) / len(df) - 0.2) < 0.02


def test_prepare_writes_files(tmp_path):
    raw_path = tmp_path / "raw.csv"
    make_raw(600, seed=4).to_csv(raw_path, index=False)
    stats = prepare(raw_path, tmp_path / "out")
    assert (tmp_path / "out" / "train.csv").exists() and (tmp_path / "out" / "stats.json").exists()
    assert stats["final_rows"] == 600
