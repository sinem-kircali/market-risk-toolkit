"""
Fetches historical daily price data via yfinance and persists it to CSV so the
rest of the toolkit can run offline. Falls back to the cached CSV in /data
when a live download isn't available or a ticker fails.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["AAPL", "JPM", "XOM", "SPY", "NVDA"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_FILENAME = "prices.csv"


def fetch_price_data(
    tickers: list[str] = None,
    start: str = "2019-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for `tickers` between `start` and `end`
    using yfinance. Returns a DataFrame indexed by date with one column per ticker.
    """
    import yfinance as yf

    tickers = tickers or DEFAULT_TICKERS

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,  # adjusted close folded into 'Close'
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError(
            "yfinance returned no data. Check your internet connection or ticker symbols."
        )

    # Normalize to a flat DataFrame of Close prices, one column per ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame({t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)})
    else:
        # Single ticker download returns a simple column index.
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index.name = "Date"
    return prices


def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data: forward-fill short gaps (holidays/halts), then drop any
    leading rows that are still NaN (ticker not yet listed), and finally drop
    any column that is still entirely empty.
    """
    cleaned = prices.sort_index().ffill()
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.dropna(how="all")
    # Drop any remaining leading NaNs (e.g. a ticker with a later IPO date)
    cleaned = cleaned.dropna(how="any")
    return cleaned


def save_price_data(prices: pd.DataFrame, filename: str = PRICES_FILENAME) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / filename
    prices.to_csv(out_path)
    return out_path


def load_price_data(filename: str = PRICES_FILENAME) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No cached price data at {path}. Run `python src/data_loader.py` first "
            "to fetch and save it."
        )
    prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    return prices


def get_price_data(
    tickers: list[str] = None,
    start: str = "2019-01-01",
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Convenience entry point: load cached CSV if present (and not `refresh`),
    otherwise fetch fresh data from yfinance, clean it, and cache it.
    """
    tickers = tickers or DEFAULT_TICKERS
    if not refresh:
        try:
            return load_price_data()
        except FileNotFoundError:
            pass

    prices = fetch_price_data(tickers, start=start, end=end)
    prices = clean_price_data(prices)
    save_price_data(prices)
    return prices


if __name__ == "__main__":
    print(f"Fetching {DEFAULT_TICKERS} from yfinance...")
    df = get_price_data(refresh=True)
    print(f"Fetched {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Saved to {DATA_DIR / PRICES_FILENAME}")
    print(df.tail())
