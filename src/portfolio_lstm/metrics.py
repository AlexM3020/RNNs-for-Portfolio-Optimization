"""Shared metrics: arithmetic annualized mean and compounded wealth/drawdown."""

import numpy as np


def wealth_and_drawdown(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Returns must be a finite nonempty vector greater than -100%.")
    wealth = np.cumprod(1 + values)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return wealth, wealth / peak - 1


def performance(returns: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(returns, dtype=float)
    wealth, drawdown = wealth_and_drawdown(values)
    mean = float(values.mean() * 252)
    vol = float(values.std(ddof=0) * np.sqrt(252))
    # Full-sample downside deviation relative to zero, not negative-only std.
    downside = float(np.sqrt(np.mean(np.minimum(values, 0) ** 2)) * np.sqrt(252))
    return {
        "observations": len(values), "annualized_return": mean,
        "annualized_volatility": vol, "sharpe": mean / vol if vol > 0 else None,
        "sortino": mean / downside if downside > 0 else None,
        "max_drawdown": float(drawdown.min()), "terminal_wealth": float(wealth[-1]),
        "positive_days_pct": float((values > 0).mean() * 100),
    }
