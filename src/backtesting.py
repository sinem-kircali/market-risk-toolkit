"""
VaR backtesting: compares a rolling VaR estimate against subsequently
realized returns and counts "breaches" (days where the actual loss exceeded
the predicted VaR). Includes the Kupiec (1995) unconditional coverage test,
the standard first check of whether a VaR model's breach rate is
statistically consistent with its stated confidence level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from var_models import historical_var


def rolling_var_forecast(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 250,
    method: str = "historical",
) -> pd.Series:
    """
    Produce a rolling out-of-sample VaR forecast: at each day t, estimate VaR
    using only the trailing `window` days of returns (no look-ahead), to be
    compared against the realized return on day t+1.
    """
    if method != "historical":
        raise NotImplementedError("Only 'historical' rolling VaR is implemented.")

    var_forecast = returns.rolling(window).apply(
        lambda w: historical_var(pd.Series(w), confidence=confidence), raw=False
    )
    return var_forecast.shift(1).dropna().rename("var_forecast")


def count_breaches(
    returns: pd.Series, var_forecast: pd.Series
) -> pd.DataFrame:
    """
    Align realized returns with the prior day's VaR forecast and flag a
    breach whenever the realized loss exceeds the forecast VaR.
    (A "loss" is -return; a breach means -return > var_forecast.)
    """
    aligned = pd.concat([returns, var_forecast], axis=1, join="inner")
    aligned.columns = ["realized_return", "var_forecast"]
    aligned["realized_loss"] = -aligned["realized_return"]
    aligned["breach"] = aligned["realized_loss"] > aligned["var_forecast"]
    return aligned


def kupiec_test(
    n_breaches: int, n_observations: int, confidence: float = 0.95
) -> dict:
    """
    Kupiec (1995) unconditional coverage (proportion of failures) test.

    Null hypothesis: the observed breach rate equals the expected failure
    rate p = 1 - confidence. Test statistic is asymptotically chi-squared
    with 1 degree of freedom.

    LR_uc = -2 * ln[ (1-p)^(n-x) * p^x / ((1 - x/n)^(n-x) * (x/n)^x) ]
    where n = n_observations, x = n_breaches, p = expected failure rate.
    """
    p = 1 - confidence
    x = n_breaches
    n = n_observations

    if n == 0:
        raise ValueError("n_observations must be > 0")

    p_hat = x / n
    expected_breaches = p * n

    # Guard against log(0) at the boundaries (x=0 or x=n).
    def _safe_log(v):
        return np.log(v) if v > 0 else np.log(1e-300)

    ll_null = (n - x) * _safe_log(1 - p) + x * _safe_log(p)
    ll_alt = (n - x) * _safe_log(1 - p_hat) + x * _safe_log(p_hat) if 0 < p_hat < 1 else ll_null

    lr_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    reject_null = p_value < 0.05

    return {
        "n_observations": n,
        "n_breaches": x,
        "expected_breaches": expected_breaches,
        "observed_breach_rate": p_hat,
        "expected_breach_rate": p,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_at_5pct": reject_null,
        "interpretation": (
            "Model REJECTED at 5% significance: breach rate is statistically "
            "inconsistent with the stated confidence level."
            if reject_null
            else "Model NOT rejected at 5% significance: breach rate is "
            "statistically consistent with the stated confidence level."
        ),
    }


def backtest_summary(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 250,
) -> dict:
    """Convenience wrapper: rolling forecast -> breaches -> Kupiec test."""
    var_forecast = rolling_var_forecast(returns, confidence=confidence, window=window)
    breaches_df = count_breaches(returns, var_forecast)
    n_breaches = int(breaches_df["breach"].sum())
    n_obs = len(breaches_df)
    kupiec = kupiec_test(n_breaches, n_obs, confidence=confidence)
    return {"breaches_df": breaches_df, "kupiec": kupiec}


if __name__ == "__main__":
    from data_loader import get_price_data
    from returns import compute_log_returns, compute_portfolio_returns

    prices = get_price_data()
    log_ret = compute_log_returns(prices)
    port_ret = compute_portfolio_returns(log_ret)

    result = backtest_summary(port_ret, confidence=0.95, window=250)
    print("Kupiec backtest (95% historical VaR, 250-day rolling window):")
    for k, v in result["kupiec"].items():
        print(f"  {k}: {v}")

    print("\nBreach days:")
    print(result["breaches_df"][result["breaches_df"]["breach"]])
