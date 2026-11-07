import numpy as np

from src.data import clean
from src.features import MODEL_INPUT_COLUMNS, add_features
from src.metrics import r2_log
from src.models import MODEL_NAMES, build_model
from tests.synthetic import make_raw


def test_every_model_trains_and_predicts_in_pkr():
    df, _ = clean(make_raw(500, seed=6))
    X, y = add_features(df)[MODEL_INPUT_COLUMNS], df["price"].to_numpy()
    for name in MODEL_NAMES:
        model = build_model(name, seed=0, n_jobs=1).fit(X[:400], y[:400])
        pred = model.predict(X[400:])
        assert pred.shape == (len(X) - 400,)
        assert np.median(pred) > 1e5, f"{name} should predict in PKR, not log units"
        assert r2_log(y[400:], pred) > 0.5, f"{name} learned nothing on structured data"
