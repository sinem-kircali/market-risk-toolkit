"""
Computes log returns per asset and builds a (configurable-weight) portfolio
return series from a price DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns: ln(P_t / P_t-1) for each column (asset)."""
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how="all")


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple (percentage) returns for each column (asset)."""
    return prices.pct_change().dropna(how="all")


def equal_weights(assets: list[str]) -> pd.Series:
    n = len(assets)
    return pd.Series(1.0 / n, index=assets)


def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series | dict | None = None,
) -> pd.Series:
    """
    Combine per-asset log returns into a single portfolio return series using
    fixed weights (rebalanced daily). Defaults to equal weighting.

    Note: summing weighted log returns is a standard small-return approximation
    for portfolio log return; it is exact for portfolio simple return.
    """
    assets = list(asset_returns.columns)
    if weights is None:
        weights = equal_weights(assets)
    elif isinstance(weights, dict):
        weights = pd.Series(weights)

    weights = weights.reindex(assets)
    if weights.isna().any():
        missing = weights[weights.isna()].index.tolist()
        raise ValueError(f"Missing weights for assets: {missing}")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.4f}")

    portfolio_returns = asset_returns.mul(weights, axis=1).sum(axis=1)
    portfolio_returns.name = "portfolio"
    return portfolio_returns


def compute_portfolio_value(
    portfolio_returns: pd.Series, initial_value: float = 100.0
) -> pd.Series:
    """Cumulative portfolio value index starting at `initial_value`."""
    cum_value = initial_value * np.exp(portfolio_returns.cumsum())
    cum_value.name = "portfolio_value"
    return cum_value


if __name__ == "__main__":
    from data_loader import get_price_data

    prices = get_price_data()
    log_ret = compute_log_returns(prices)
    port_ret = compute_portfolio_returns(log_ret)
    port_val = compute_portfolio_value(port_ret)

    print("Asset log returns (tail):")
    print(log_ret.tail())
    print("\nEqual-weighted portfolio returns (tail):")
    print(port_ret.tail())
    print("\nPortfolio value index (tail):")
    print(port_val.tail())
