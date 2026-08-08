"""Basic unit tests for src/var_models.py and src/expected_shortfall.py."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from var_models import historical_var, parametric_var, monte_carlo_var, var_summary
from expected_shortfall import historical_es, parametric_es, monte_carlo_es


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.01, size=2000))


@pytest.fixture
def known_returns():
    # Deterministic series: 1% steps from -5% to +4%, easy to reason about quantiles.
    return pd.Series(np.arange(-50, 50) / 1000.0)


def test_historical_var_is_positive(normal_returns):
    var95 = historical_var(normal_returns, confidence=0.95)
    assert var95 > 0


def test_parametric_var_is_positive(normal_returns):
    var95 = parametric_var(normal_returns, confidence=0.95)
    assert var95 > 0


def test_monte_carlo_var_is_positive(normal_returns):
    var95 = monte_carlo_var(normal_returns, confidence=0.95, n_simulations=5000)
    assert var95 > 0


def test_99pct_var_exceeds_95pct_var(normal_returns):
    var95 = historical_var(normal_returns, confidence=0.95)
    var99 = historical_var(normal_returns, confidence=0.99)
    assert var99 >= var95

    var95_p = parametric_var(normal_returns, confidence=0.95)
    var99_p = parametric_var(normal_returns, confidence=0.99)
    assert var99_p >= var95_p


def test_historical_var_matches_known_quantile(known_returns):
    # 5th percentile of a uniform ramp from -0.05 to 0.049 should be close to -0.045
    var95 = historical_var(known_returns, confidence=0.95)
    assert var95 == pytest.approx(0.0455, abs=0.01)


def test_parametric_var_matches_normal_formula():
    returns = pd.Series([0.0] * 1000)  # mu=0, sigma=0 -> VaR should be 0
    assert parametric_var(returns, confidence=0.95) == 0.0


def test_monte_carlo_var_converges_to_parametric(normal_returns):
    # With a Normal fit and enough simulations, MC VaR should be close to
    # the closed-form parametric VaR.
    var_param = parametric_var(normal_returns, confidence=0.95)
    var_mc = monte_carlo_var(normal_returns, confidence=0.95, n_simulations=50_000, seed=1)
    assert var_mc == pytest.approx(var_param, rel=0.15)


def test_var_summary_shape(normal_returns):
    summary = var_summary(normal_returns, confidences=[0.95, 0.99])
    assert list(summary.index) == [0.95, 0.99]
    assert set(summary.columns) == {"historical_var", "parametric_var", "monte_carlo_var"}
    assert (summary >= 0).all().all()


def test_expected_shortfall_exceeds_var(normal_returns):
    var95 = historical_var(normal_returns, confidence=0.95)
    es95 = historical_es(normal_returns, confidence=0.95)
    assert es95 >= var95

    var95_p = parametric_var(normal_returns, confidence=0.95)
    es95_p = parametric_es(normal_returns, confidence=0.95)
    assert es95_p >= var95_p


def test_monte_carlo_es_is_positive(normal_returns):
    es95 = monte_carlo_es(normal_returns, confidence=0.95, n_simulations=5000)
    assert es95 > 0


def test_var_scales_with_horizon(normal_returns):
    var_1day = parametric_var(normal_returns, confidence=0.95, horizon=1)
    var_10day = parametric_var(normal_returns, confidence=0.95, horizon=10)
    assert var_10day > var_1day
