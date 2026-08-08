# Market Risk Toolkit

A market risk analysis toolkit built on a real 5-asset equity portfolio (AAPL, JPM, XOM, SPY, NVDA),
implementing the core measures used on a trading desk or in a market risk function: Value-at-Risk
(historical, parametric, and Monte Carlo), Expected Shortfall / CVaR, scenario-based stress testing,
regime detection via a 2-state Hidden Markov Model, and out-of-sample VaR backtesting with the Kupiec
unconditional coverage test. It's built to demonstrate the practical, quantitative skill set behind
market/trading risk analyst work — not just data science — using real historical price data (via
`yfinance`), a reproducible analysis notebook, and an interactive Streamlit dashboard for exploring
the results across different tickers, date ranges, and portfolio weights.

## Project structure

```
market-risk-toolkit/
├── data/                        # cached historical price CSV (so the project runs offline)
├── src/
│   ├── data_loader.py           # fetches & caches historical prices via yfinance
│   ├── returns.py                # log returns, portfolio return series
│   ├── var_models.py             # Historical / Parametric / Monte Carlo VaR
│   ├── expected_shortfall.py     # Expected Shortfall (CVaR)
│   ├── stress_testing.py         # equity shock, vol spike, correlation breakdown scenarios
│   ├── regime_detection.py       # 2-state Gaussian HMM (Baum-Welch) for calm/volatile regimes
│   └── backtesting.py            # rolling VaR backtest + Kupiec test
├── notebooks/
│   └── 01_full_analysis.ipynb    # end-to-end walkthrough with charts & commentary
├── app/
│   └── dashboard.py               # interactive Streamlit dashboard
└── tests/
    └── test_var_models.py         # unit tests for the VaR/ES functions
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Refresh the data (optional)

The repo ships with `data/prices.csv` already fetched, so nothing below requires a live connection.
To pull fresh data:

```bash
python src/data_loader.py
```

### Run the notebook

```bash
jupyter notebook notebooks/01_full_analysis.ipynb
```

### Run the dashboard

```bash
streamlit run app/dashboard.py
```

### Run the tests

```bash
pytest tests/ -v
```

## Risk measures, explained

**Value-at-Risk (VaR)** answers "what's the most I'd expect to lose over one day, X% of the
time?" This project computes it three ways: **historical simulation** (the empirical quantile of
past returns — makes no distributional assumption but is entirely backward-looking), **parametric
/ variance-covariance** (assumes returns are Normally distributed — fast and analytically clean,
but tends to understate tail risk since real markets have fatter tails than a Normal
distribution), and **Monte Carlo** (simulates thousands of return paths from a fitted distribution
and takes the empirical quantile of the simulated outcomes).

**Expected Shortfall (ES / CVaR)** goes one step further than VaR: instead of just the threshold
loss at a given confidence level, it's the *average* loss in the scenarios where that threshold
is breached. It's a more conservative, "coherent" risk measure — one of the reasons Basel III/FRTB
now favors ES over VaR for regulatory capital.

**Stress testing** asks "what happens to the portfolio under a specific hypothetical shock?" —
independent of what actually happened historically. This project runs three scenarios: a uniform
equity shock (all assets -15%), a volatility spike (VaR/ES recomputed with volatility doubled),
and a correlation breakdown / contagion scenario (all pairwise correlations forced to 1.0, which
wipes out diversification benefit entirely — the classic 2008/March-2020 failure mode).

**Regime detection** classifies each trading day as "calm" or "volatile" using a 2-state Gaussian
Hidden Markov Model fit (via Baum-Welch/EM, implemented from scratch) on rolling realized
volatility. It also reports the day-to-day transition probabilities between regimes, which is
useful context for risk models that behave differently depending on the prevailing market regime.

## Methodology & assumptions

- **Parametric and Monte Carlo VaR/ES assume Normally distributed returns.** Real equity returns
  exhibit excess kurtosis (fat tails) and some skew, so these methods will typically understate
  tail risk relative to historical simulation, especially at the 99% confidence level. This is
  visible directly in the notebook's VaR comparison chart.
- **Historical VaR/ES is fully backward-looking** and assumes the future tail distribution
  resembles the sample period. A regime or shock outside the historical window (e.g. a novel
  crisis) won't be reflected until it happens — which is exactly why stress testing is run
  alongside it rather than as a replacement.
- **Portfolio log returns are combined as weight-weighted sums of asset log returns**, a standard
  small-return approximation that is exact for simple (percentage) portfolio returns and very
  close for daily log returns.
- **The regime detection model is unsupervised** — it labels the lower-mean-volatility state
  "calm" and the higher one "volatile" after fitting, with no hand-labeled ground truth. It is
  a descriptive/exploratory tool, not a trading signal.
- **The Kupiec test only checks unconditional coverage** (i.e. the overall breach rate), not
  whether breaches cluster in time (that would require a conditional coverage test, e.g.
  Christoffersen). A model can pass Kupiec while still failing to capture volatility clustering.
- Data is daily adjusted close prices from Yahoo Finance via `yfinance`, 2019-present, covering
  both the 2020 COVID crash and the 2022 rate-hike selloff as in-sample stress periods.

## Screenshots

_Add screenshots here after running the dashboard locally, e.g.:_

```markdown
![Portfolio overview](docs/screenshot_overview.png)
![VaR & Expected Shortfall](docs/screenshot_var_es.png)
![Stress testing](docs/screenshot_stress.png)
![Regime detection](docs/screenshot_regimes.png)
```

## Disclaimer

Built as a portfolio project for educational purposes. Not investment advice, and not intended
for use in live trading or capital-allocation decisions.
