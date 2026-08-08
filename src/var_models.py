"""
Value-at-Risk (VaR) estimation using three standard approaches:

  1. Historical simulation      - empirical quantile of realized returns
  2. Parametric / variance-covariance - assumes normally distributed returns
  3. Monte Carlo simulation     - simulates return paths from a fitted normal
                                  distribution

All functions report VaR as a *positive* number representing the loss
magnitude (e.g. 0.025 = a 2.5% loss) at the given confidence level, over a
1-day horizon unless `horizon` is scaled.

Confidence conventions: confidence=0.95 means "95% confident daily loss will
not exceed VaR" i.e. we look at the 5th percentile of the return distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """
    Historical simulation VaR: empirical quantile of the historical return
    distribution, scaled to the horizon by sqrt(time).

    Pros: makes no distributional assumption, captures real fat tails/skew.
    Cons: entirely backward-looking; a shock that never occurred in the
    sample won't show up.
    """
    alpha = 1 - confidence
    quantile_return = np.percentile(returns.dropna(), alpha * 100)
    var_1day = -quantile_return
    return max(var_1day * np.sqrt(horizon), 0.0)


def parametric_var(
    returns: pd.Series, confidence: float = 0.95, horizon: int = 1
) -> float:
    """
    Parametric (variance-covariance) VaR assuming returns ~ Normal(mu, sigma).

    VaR = -(mu + z_alpha * sigma), scaled to horizon by sqrt(time) for sigma
    and linearly for mu.

    Pros: fast, smooth, easy to decompose analytically.
    Cons: real returns exhibit fat tails and skew that Normal understates,
    so this tends to underestimate tail risk in stressed markets.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)  # negative value, e.g. -1.645 for 95%
    var_1day = -(mu + z * sigma)
    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)
    var_h = -(mu_h + z * sigma_h)
    return max(var_h, 0.0)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> float:
    """
    Monte Carlo VaR: fit a Normal distribution to historical returns, simulate
    `n_simulations` random daily-return paths over `horizon` days, and take
    the empirical quantile of simulated cumulative returns.

    Pros: extends naturally to more complex distributions/multi-asset
    correlation structures.
    Cons: only as good as the fitted distribution - here still Normal, so it
    inherits the same thin-tail limitation as parametric VaR plus simulation
    noise.
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean()
    sigma = returns.std()

    simulated_daily = rng.normal(mu, sigma, size=(n_simulations, horizon))
    simulated_cumulative = simulated_daily.sum(axis=1)

    alpha = 1 - confidence
    quantile_return = np.percentile(simulated_cumulative, alpha * 100)
    return max(-quantile_return, 0.0)


def var_summary(
    returns: pd.Series,
    confidences: list[float] = (0.95, 0.99),
    horizon: int = 1,
    n_simulations: int = 10_000,
) -> pd.DataFrame:
    """Build a tidy table of all three VaR methods x all confidence levels."""
    rows = []
    for conf in confidences:
        rows.append(
            {
                "confidence": conf,
                "historical_var": historical_var(returns, conf, horizon),
                "parametric_var": parametric_var(returns, conf, horizon),
                "monte_carlo_var": monte_carlo_var(
                    returns, conf, horizon, n_simulations
                ),
            }
        )
    return pd.DataFrame(rows).set_index("confidence")


if __name__ == "__main__":
    from data_loader import get_price_data
    from returns import compute_log_returns, compute_portfolio_returns

    prices = get_price_data()
    log_ret = compute_log_returns(prices)
    port_ret = compute_portfolio_returns(log_ret)

    summary = var_summary(port_ret)
    print("1-day VaR summary (fraction of portfolio value):")
    print(summary)
    print("\nAs dollar loss on a $1,000,000 portfolio:")
    print((summary * 1_000_000).round(0))
