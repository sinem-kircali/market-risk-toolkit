"""
Streamlit dashboard for the Market Risk Toolkit.

Run with:
    streamlit run app/dashboard.py

Lets a user pick tickers and a date range, then view portfolio value/returns,
VaR & ES at 95%/99%, a triggered stress scenario, and a regime chart overlaid
on the price series.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import DEFAULT_TICKERS, get_price_data
from returns import compute_log_returns, compute_portfolio_returns, compute_portfolio_value, equal_weights
from var_models import var_summary
from expected_shortfall import es_summary
from stress_testing import equity_shock, volatility_spike_var, correlation_breakdown
from regime_detection import classify_regimes

st.set_page_config(page_title="Market Risk Toolkit", layout="wide")

st.title("Market Risk Toolkit")
st.caption(
    "Historical / Parametric / Monte Carlo VaR, Expected Shortfall, stress testing, "
    "and Markov regime detection on a real equity portfolio."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Portfolio configuration")

all_prices = get_price_data()  # loads cached CSV; falls back to yfinance if missing

available_tickers = list(all_prices.columns)
selected_tickers = st.sidebar.multiselect(
    "Tickers", options=available_tickers, default=available_tickers
)
if len(selected_tickers) < 2:
    st.warning("Select at least 2 tickers.")
    st.stop()

min_date = all_prices.index.min().date()
max_date = all_prices.index.max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if len(date_range) != 2:
    st.stop()
start_date, end_date = date_range

weighting = st.sidebar.radio("Weighting", ["Equal-weighted", "Custom"])
if weighting == "Custom":
    st.sidebar.write("Weights (must sum to 1.0):")
    custom_weights = {}
    for t in selected_tickers:
        custom_weights[t] = st.sidebar.number_input(
            t, min_value=0.0, max_value=1.0, value=round(1 / len(selected_tickers), 2), step=0.05
        )
    weight_sum = sum(custom_weights.values())
    if not np.isclose(weight_sum, 1.0, atol=0.01):
        st.sidebar.error(f"Weights sum to {weight_sum:.2f}, must be ~1.0")
        st.stop()
    weights = pd.Series(custom_weights)
else:
    weights = equal_weights(selected_tickers)

window = st.sidebar.slider("Regime detection rolling window (days)", 10, 60, 21)

# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
prices = all_prices.loc[str(start_date):str(end_date), selected_tickers]
log_returns = compute_log_returns(prices)
portfolio_returns = compute_portfolio_returns(log_returns, weights)
portfolio_value = compute_portfolio_value(portfolio_returns)

# ---------------------------------------------------------------------------
# Portfolio value / returns chart
# ---------------------------------------------------------------------------
st.subheader("Portfolio value")
col1, col2 = st.columns([2, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value.values, name="Portfolio value (base=100)"))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    ann_return = portfolio_returns.mean() * 252
    st.metric("Total return (period)", f"{total_return:.1%}")
    st.metric("Annualized volatility", f"{ann_vol:.1%}")
    st.metric("Annualized return", f"{ann_return:.1%}")
    st.write("Weights:")
    st.dataframe(weights.rename("weight").to_frame())

# ---------------------------------------------------------------------------
# VaR & ES
# ---------------------------------------------------------------------------
st.subheader("Value at Risk & Expected Shortfall (1-day)")
var_table = var_summary(portfolio_returns, confidences=[0.95, 0.99])
es_table = es_summary(portfolio_returns, confidences=[0.95, 0.99])

c1, c2 = st.columns(2)
with c1:
    st.markdown("**VaR (fraction of portfolio value)**")
    st.dataframe(var_table.style.format("{:.2%}"))
with c2:
    st.markdown("**Expected Shortfall / CVaR (fraction of portfolio value)**")
    st.dataframe(es_table.style.format("{:.2%}"))

portfolio_notional = st.number_input("Portfolio notional ($) for dollar-loss estimates", value=1_000_000, step=100_000)
st.dataframe((var_table * portfolio_notional).round(0).rename(columns=lambda c: c + "_$"))

# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------
st.subheader("Stress testing")
scenario = st.selectbox(
    "Scenario",
    ["Equity shock: all assets -15%", "Volatility doubles", "Correlation breakdown (contagion, rho=1)"],
)

if scenario == "Equity shock: all assets -15%":
    shock_pct = st.slider("Shock size", -0.50, -0.01, -0.15, step=0.01, format="%.2f")
    impact = equity_shock(weights, shock_pct)
    st.metric("Portfolio impact", f"{impact:.2%}", delta=f"${impact * portfolio_notional:,.0f}")
elif scenario == "Volatility doubles":
    multiplier = st.slider("Volatility multiplier", 1.0, 5.0, 2.0, step=0.5)
    result = volatility_spike_var(portfolio_returns, multiplier=multiplier, confidence=0.99)
    c1, c2 = st.columns(2)
    c1.metric("Baseline 99% VaR", f"{result['baseline_var']:.2%}")
    c2.metric(f"Shocked 99% VaR ({multiplier:.1f}x vol)", f"{result['shocked_var']:.2%}",
               delta=f"{result['shocked_var'] - result['baseline_var']:.2%}")
else:
    result = correlation_breakdown(log_returns, weights)
    c1, c2 = st.columns(2)
    c1.metric("Portfolio vol (actual correlation)", f"{result['portfolio_vol_actual']:.2%}")
    c2.metric("Portfolio vol (contagion, rho=1)", f"{result['portfolio_vol_contagion']:.2%}",
               delta=f"{result['vol_increase_pct']:.1%}")
    st.caption("Diversification benefit disappears when every asset moves together.")

# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------
st.subheader("Market regime detection (2-state Markov model)")
regimes, transmat, params = classify_regimes(portfolio_returns, window=window)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
price_index = compute_portfolio_value(portfolio_returns)
fig2.add_trace(
    go.Scatter(x=price_index.index, y=price_index.values, name="Portfolio value", line=dict(color="royalblue")),
    secondary_y=False,
)
volatile_mask = regimes["regime"] == "volatile"
fig2.add_trace(
    go.Scatter(
        x=regimes.index[volatile_mask],
        y=price_index.reindex(regimes.index)[volatile_mask],
        mode="markers",
        marker=dict(color="crimson", size=4),
        name="Volatile regime",
    ),
    secondary_y=False,
)
fig2.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig2, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Mean annualized vol (calm)", f"{params['mean_vol_calm']:.1%}")
c2.metric("Mean annualized vol (volatile)", f"{params['mean_vol_volatile']:.1%}")
c3.metric("Days classified volatile", f"{volatile_mask.sum()} / {len(regimes)}")

st.markdown("**Transition matrix** (probability of moving from row-state to column-state next day)")
st.dataframe(params["transition_matrix"].style.format("{:.2%}"))

st.caption(
    "Built as a portfolio project demonstrating market risk analysis: VaR, Expected Shortfall, "
    "stress testing, backtesting, and Markov regime detection on real market data."
)
