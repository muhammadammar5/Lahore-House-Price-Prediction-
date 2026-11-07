import numpy as np

from src.metrics import median_ape, r2_log, regression_report, rmse, rmsle


def test_perfect_prediction():
    y = np.array([1e7, 2e7, 5e7])
    rep = regression_report(y, y)
    assert rep["rmse_pkr"] == 0 and rep["r2"] == 1.0 and rep["median_ape_pct"] == 0


def test_rmse_and_ape_known_values():
    y, p = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert np.isclose(rmse(y, p), np.sqrt((100 + 400) / 2))
    assert np.isclose(median_ape(y, p), 0.10)


def test_log_metrics_are_scale_free():
    y = np.array([1e6, 1e8])
    assert np.isclose(rmsle(y, y * 1.1), rmsle(y * 10, y * 10 * 1.1), atol=1e-4)
    assert r2_log(y, y) == 1.0


def test_negative_predictions_are_clipped():
    assert np.isfinite(rmsle(np.array([1e6]), np.array([-5.0])))
