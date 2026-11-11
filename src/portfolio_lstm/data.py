"""Date-preserving price loading and strictly next-day sequence targets."""

from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = ("VTI", "AGG", "DBC", "VIX")


def load_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices.rename(columns={"^VIX": "VIX"})
    if set(prices.columns) != set(ASSETS):
        raise ValueError(f"Expected price columns {ASSETS}; received {tuple(prices.columns)}.")
    prices = prices.loc[:, list(ASSETS)].astype(float)
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Prices need valid calendar dates.")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Prices must be positive and finite; missing rows require explicit handling.")
    return prices


def make_samples(prices: pd.DataFrame, lookback: int = 50):
    """Input ends at t-1; target and recorded date are the realized return at t."""
    if lookback < 2 or len(prices) <= lookback + 2:
        raise ValueError("Insufficient observations for the requested lookback.")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    features = np.column_stack((prices.iloc[1:].to_numpy(), returns.to_numpy()))
    x = np.stack([features[t - lookback:t] for t in range(lookback, len(features))])
    return x.astype(np.float32), returns.iloc[lookback:].copy()


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit on training windows only; constant features retain unit scale."""
    mean = x.mean(axis=(0, 1), dtype=np.float64)
    scale = x.std(axis=(0, 1), dtype=np.float64)
    return mean, np.where(scale > 1e-8, scale, 1.0)


def download_prices(output: Path, start: str, end: str) -> None:
    """Explicit-only Yahoo Finance download; end date is exclusive."""
    import yfinance as yf

    frame = yf.download(["VTI", "AGG", "DBC", "^VIX"], start=start, end=end,
                        auto_adjust=True, progress=False)["Close"]
    frame = frame.rename(columns={"^VIX": "VIX"}).dropna()
    if frame.empty:
        raise ValueError("No complete price observations returned by Yahoo Finance.")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(ASSETS)].to_csv(output)
    load_prices(output)
