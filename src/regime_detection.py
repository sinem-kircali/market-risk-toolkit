"""
Regime detection via a 2-state Gaussian Hidden Markov Model fit (with the
Baum-Welch / EM algorithm, implemented from scratch on numpy/scipy) to
rolling realized volatility. The two hidden states are interpreted as:

    - "calm"     : the state with the lower fitted mean volatility
    - "volatile" : the state with the higher fitted mean volatility

This gives both a day-by-day regime classification (posterior-probability
argmax) and the underlying 2x2 Markov transition matrix, i.e. the
probability of staying calm/volatile vs switching regimes tomorrow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_rolling_volatility(
    returns: pd.Series, window: int = 21, annualize: bool = True
) -> pd.Series:
    """Rolling realized volatility of daily returns, optionally annualized."""
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol.dropna().rename("rolling_vol")


class TwoStateGaussianHMM:
    """
    A minimal 2-state Gaussian Hidden Markov Model estimated via Baum-Welch.

    Parameters
    ----------
    n_iter : maximum EM iterations
    tol    : stop early once the log-likelihood improves by less than `tol`
    seed   : RNG seed for initialization
    """

    def __init__(self, n_iter: int = 200, tol: float = 1e-6, seed: int = 42):
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None
        self.loglik_history_: list[float] = []

    def _emission_probs(self, obs: np.ndarray) -> np.ndarray:
        B = np.column_stack(
            [norm.pdf(obs, self.means_[k], np.sqrt(self.vars_[k])) for k in range(2)]
        )
        return np.clip(B, 1e-300, None)

    def _forward_backward(self, obs: np.ndarray):
        T = len(obs)
        B = self._emission_probs(obs)

        alpha = np.zeros((T, 2))
        c = np.zeros(T)
        alpha[0] = self.startprob_ * B[0]
        c[0] = alpha[0].sum() or 1e-300
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * B[t]
            c[t] = alpha[t].sum() or 1e-300
            alpha[t] /= c[t]

        beta = np.zeros((T, 2))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.transmat_ @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T - 1, 2, 2))
        for t in range(T - 1):
            xi[t] = (
                alpha[t][:, None] * self.transmat_ * B[t + 1][None, :] * beta[t + 1][None, :]
            ) / c[t + 1]

        loglik = np.sum(np.log(c))
        return gamma, xi, loglik

    def fit(self, obs: np.ndarray) -> "TwoStateGaussianHMM":
        obs = np.asarray(obs, dtype=float)
        rng = np.random.default_rng(self.seed)

        median = np.median(obs)
        low, high = obs[obs <= median], obs[obs > median]
        self.means_ = np.array([low.mean(), high.mean()])
        self.vars_ = np.array([low.var() + 1e-8, high.var() + 1e-8])
        self.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
        self.startprob_ = np.array([0.5, 0.5])

        prev_loglik = -np.inf
        for _ in range(self.n_iter):
            gamma, xi, loglik = self._forward_backward(obs)
            self.loglik_history_.append(loglik)

            self.startprob_ = gamma[0].copy()
            xi_sum = xi.sum(axis=0)
            denom = gamma[:-1].sum(axis=0)
            self.transmat_ = xi_sum / denom[:, None]

            for k in range(2):
                w = gamma[:, k]
                w_sum = w.sum()
                self.means_[k] = (w * obs).sum() / w_sum
                self.vars_[k] = (w * (obs - self.means_[k]) ** 2).sum() / w_sum
                self.vars_[k] = max(self.vars_[k], 1e-8)

            if abs(loglik - prev_loglik) < self.tol:
                break
            prev_loglik = loglik

        return self

    def predict_proba(self, obs: np.ndarray) -> np.ndarray:
        gamma, _, _ = self._forward_backward(np.asarray(obs, dtype=float))
        return gamma

    def predict_states(self, obs: np.ndarray) -> np.ndarray:
        return self.predict_proba(obs).argmax(axis=1)


def classify_regimes(
    returns: pd.Series, window: int = 21, n_iter: int = 200, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Fit a 2-state Gaussian HMM on rolling volatility of `returns` and return:

      - a DataFrame indexed by date with columns [rolling_vol, state, regime,
        prob_calm, prob_volatile]
      - the 2x2 transition matrix, ordered [calm, volatile]
      - a dict of fitted params (means/vars per regime, for reporting)
    """
    vol = compute_rolling_volatility(returns, window=window)

    hmm = TwoStateGaussianHMM(n_iter=n_iter, seed=seed).fit(vol.values)
    gamma = hmm.predict_proba(vol.values)
    raw_states = gamma.argmax(axis=1)

    # Relabel so state 0 = "calm" (lower mean vol), state 1 = "volatile"
    calm_idx = int(np.argmin(hmm.means_))
    volatile_idx = 1 - calm_idx
    label_map = {calm_idx: "calm", volatile_idx: "volatile"}
    regime_labels = np.array([label_map[s] for s in raw_states])

    prob_calm = gamma[:, calm_idx]
    prob_volatile = gamma[:, volatile_idx]

    result = pd.DataFrame(
        {
            "rolling_vol": vol.values,
            "state": raw_states,
            "regime": regime_labels,
            "prob_calm": prob_calm,
            "prob_volatile": prob_volatile,
        },
        index=vol.index,
    )

    # Reorder the transition matrix so row/col 0 = calm, 1 = volatile
    order = [calm_idx, volatile_idx]
    transmat_ordered = hmm.transmat_[np.ix_(order, order)]

    params = {
        "mean_vol_calm": hmm.means_[calm_idx],
        "mean_vol_volatile": hmm.means_[volatile_idx],
        "std_vol_calm": np.sqrt(hmm.vars_[calm_idx]),
        "std_vol_volatile": np.sqrt(hmm.vars_[volatile_idx]),
        "transition_matrix": pd.DataFrame(
            transmat_ordered,
            index=["from_calm", "from_volatile"],
            columns=["to_calm", "to_volatile"],
        ),
    }

    return result, transmat_ordered, params


if __name__ == "__main__":
    from data_loader import get_price_data
    from returns import compute_log_returns, compute_portfolio_returns

    prices = get_price_data()
    log_ret = compute_log_returns(prices)
    port_ret = compute_portfolio_returns(log_ret)

    regimes, transmat, params = classify_regimes(port_ret)

    print("Regime classification (tail):")
    print(regimes.tail(10))
    print(f"\nMean annualized vol - calm: {params['mean_vol_calm']:.2%}, "
          f"volatile: {params['mean_vol_volatile']:.2%}")
    print("\nTransition matrix:")
    print(params["transition_matrix"])
    print(f"\nDays classified calm: {(regimes['regime'] == 'calm').sum()}")
    print(f"Days classified volatile: {(regimes['regime'] == 'volatile').sum()}")
