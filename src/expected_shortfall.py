"""
Expected Shortfall (ES), also known as Conditional VaR (CVaR): the average
loss in the tail beyond the VaR threshold. Where VaR answers "what's the
worst loss I should expect 95% of the time?", ES answers "given that we're
in that bad 5% of days, how bad is it on average?" ES is a coherent risk
measure (subadditive) and is the metric Basel III/FRTB now favors over VaR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_es(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """
    Historical ES: average of realized returns at or below the historical
    VaR quantile (the empirical tail mean).
    """
    alpha = 1 - confidence
    threshold = np.percentile(returns.dropna(), alpha * 100)
    tail_losses = returns[returns <= threshold]
    if len(tail_losses) == 0:
        tail_losses = pd.Series([threshold])
    es_1day = -tail_losses.mean()
    return max(es_1day * np.sqrt(horizon), 0.0)


def parametric_es(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """
    Parametric ES under a Normal assumption, closed form:
        ES = -(mu - sigma * phi(z_alpha) / alpha)
    where phi is the standard normal pdf and z_alpha = Phi^-1(1 - confidence).
    """
    mu = returns.mean() * horizon
    sigma = returns.std() * np.sqrt(horizon)
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    es = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return max(es, 0.0)


def monte_carlo_es(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> float:
    """Monte Carlo ES: simulate paths, then average the losses beyond the VaR quantile."""
    rng = np.random.default_rng(seed)
    mu = returns.mean()
    sigma = returns.std()

    simulated_daily = rng.normal(mu, sigma, size=(n_simulations, horizon))
    simulated_cumulative = simulated_daily.sum(axis=1)

    alpha = 1 - confidence
    threshold = np.percentile(simulated_cumulative, alpha * 100)
    tail = simulated_cumulative[simulated_cumulative <= threshold]
    return max(-tail.mean(), 0.0)


def es_summary(
    returns: pd.Series,
    confidences: list[float] = (0.95, 0.99),
    horizon: int = 1,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    rows = []
    for conf in confidences:
        rows.append(
            {
                "confidence": conf,
                "historical_es": historical_es(returns, conf, horizon),
                "parametric_es": parametric_es(returns, conf, horizon),
                "monte_carlo_es": monte_carlo_es(returns, conf, horizon, n_simulations),
            }
        )
    return pd.DataFrame(rows).set_index("confidence")


if __name__ == "__main__":
    from data_loader import get_price_data
    from returns import compute_log_returns, compute_portfolio_returns

    prices = get_price_data()
    log_ret = compute_log_returns(prices)
    port_ret = compute_portfolio_returns(log_ret)

    summary = es_summary(port_ret)
    print("1-day Expected Shortfall summary (fraction of portfolio value):")
    print(summary)
