"""
Hypothetical scenario stress testing. Unlike VaR/ES (which are backward- or
distribution-derived), stress tests ask "what happens to the portfolio under
a specific, analyst-defined shock?" - useful for tail scenarios that may be
absent or under-represented in the historical sample.

Three scenarios are implemented:
  1. Equity shock       - every asset drops by a fixed percentage simultaneously
  2. Volatility spike   - daily volatility doubles (or any multiplier), VaR/ES recomputed
  3. Correlation breakdown / contagion - all pairwise correlations go to 1.0,
     removing diversification benefit entirely
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def equity_shock(
    weights: pd.Series, shock_pct: float = -0.15
) -> float:
    """
    Apply a uniform simultaneous price shock (e.g. -0.15 = -15%) to every
    asset and return the resulting portfolio return (loss if negative).
    Since the shock is applied uniformly, the portfolio impact equals the
    shock itself regardless of weights - shown explicitly here for clarity
    and to support non-uniform per-asset shocks if extended later.
    """
    per_asset_shock = pd.Series(shock_pct, index=weights.index)
    portfolio_impact = (weights * per_asset_shock).sum()
    return portfolio_impact


def asymmetric_shock(weights: pd.Series, shocks: dict) -> float:
    """
    Apply a different shock to each asset, e.g. {"XOM": -0.25, "AAPL": -0.10}.
    Assets not present in `shocks` are assumed unshocked (0.0).
    """
    per_asset_shock = pd.Series(0.0, index=weights.index)
    for ticker, shock in shocks.items():
        if ticker in per_asset_shock.index:
            per_asset_shock[ticker] = shock
    return (weights * per_asset_shock).sum()


def volatility_spike_var(
    returns: pd.Series,
    multiplier: float = 2.0,
    confidence: float = 0.95,
    horizon: int = 1,
) -> dict:
    """
    Recompute parametric VaR/ES assuming daily volatility is `multiplier`x
    the historical estimate (mean return held fixed). Returns baseline vs
    shocked VaR/ES so the delta is explicit.
    """
    mu = returns.mean() * horizon
    sigma = returns.std() * np.sqrt(horizon)
    sigma_shocked = sigma * multiplier

    z = stats.norm.ppf(1 - confidence)
    var_base = max(-(mu + z * sigma), 0.0)
    var_shocked = max(-(mu + z * sigma_shocked), 0.0)

    alpha = 1 - confidence
    z_es = stats.norm.ppf(alpha)
    es_base = max(-(mu - sigma * stats.norm.pdf(z_es) / alpha), 0.0)
    es_shocked = max(-(mu - sigma_shocked * stats.norm.pdf(z_es) / alpha), 0.0)

    return {
        "baseline_var": var_base,
        "shocked_var": var_shocked,
        "baseline_es": es_base,
        "shocked_es": es_shocked,
        "multiplier": multiplier,
    }


def correlation_breakdown(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    horizon: int = 1,
) -> dict:
    """
    Compare portfolio volatility under the actual (historical) correlation
    matrix vs a contagion scenario where all pairwise correlations jump to
    1.0 (every asset moves together - the diversification benefit vanishes).

    Portfolio variance = w' Sigma w, where Sigma = D R D, D = diag(sigma_i),
    R = correlation matrix.
    """
    weights = weights.reindex(asset_returns.columns)
    sigmas = asset_returns.std().values
    w = weights.values

    corr_actual = asset_returns.corr().values
    cov_actual = np.outer(sigmas, sigmas) * corr_actual
    portfolio_var_actual = w @ cov_actual @ w
    portfolio_vol_actual = np.sqrt(max(portfolio_var_actual, 0.0)) * np.sqrt(horizon)

    corr_stressed = np.ones_like(corr_actual)
    cov_stressed = np.outer(sigmas, sigmas) * corr_stressed
    portfolio_var_stressed = w @ cov_stressed @ w
    portfolio_vol_stressed = np.sqrt(max(portfolio_var_stressed, 0.0)) * np.sqrt(horizon)

    return {
        "portfolio_vol_actual": portfolio_vol_actual,
        "portfolio_vol_contagion": portfolio_vol_stressed,
        "vol_increase_pct": (portfolio_vol_stressed / portfolio_vol_actual - 1)
        if portfolio_vol_actual > 0
        else np.nan,
    }


def run_stress_tests(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.Series,
    weights: pd.Series,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """Run all three scenarios and return a tidy summary table."""
    rows = []

    eq_shock = equity_shock(weights, -0.15)
    rows.append(
        {
            "scenario": "Equity shock: all assets -15%",
            "metric": "Portfolio return",
            "value": eq_shock,
        }
    )

    vol_result = volatility_spike_var(portfolio_returns, multiplier=2.0, confidence=confidence)
    rows.append(
        {
            "scenario": "Volatility doubles",
            "metric": f"VaR {int(confidence*100)}% (baseline -> shocked)",
            "value": vol_result["shocked_var"] - vol_result["baseline_var"],
        }
    )

    corr_result = correlation_breakdown(asset_returns, weights)
    rows.append(
        {
            "scenario": "Correlation breakdown (contagion, rho=1)",
            "metric": "Portfolio volatility increase",
            "value": corr_result["vol_increase_pct"],
        }
    )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import get_price_data
    from returns import compute_log_returns, compute_portfolio_returns, equal_weights

    prices = get_price_data()
    log_ret = compute_log_returns(prices)
    weights = equal_weights(list(prices.columns))
    port_ret = compute_portfolio_returns(log_ret, weights)

    print("=== Equity shock -15% ===")
    print(f"Portfolio impact: {equity_shock(weights, -0.15):.2%}")

    print("\n=== Volatility doubles ===")
    for k, v in volatility_spike_var(port_ret).items():
        print(f"{k}: {v}")

    print("\n=== Correlation breakdown (contagion) ===")
    for k, v in correlation_breakdown(log_ret, weights).items():
        print(f"{k}: {v}")

    print("\n=== Summary table ===")
    print(run_stress_tests(log_ret, port_ret, weights))
