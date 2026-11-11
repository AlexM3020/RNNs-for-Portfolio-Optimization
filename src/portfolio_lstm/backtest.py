"""Refactored expanding-window experiment; not a replay of archived numbers."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .baselines import optimized_weights
from .data import ASSETS, fit_scaler, load_prices, make_samples
from .metrics import performance
from .model import PortfolioLSTM, sharpe_ratio_loss


@dataclass
class Settings:
    lookback: int = 50
    hidden_size: int = 128
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0001
    patience: int = 20
    validation_fraction: float = 0.1
    seed: int = 42
    first_test_year: int = 2012
    step_years: int = 2


def run_backtest(prices_path: Path, output: Path, settings: Settings) -> dict:
    """Write dated, untouched-OOS predictions and separately selected fold models."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory; existing runs are never overwritten.")
    if min(settings.epochs, settings.hidden_size, settings.patience, settings.step_years) < 1:
        raise ValueError("Epochs, hidden size, patience and year step must be positive.")
    if settings.batch_size < 2 or not 0 < settings.validation_fraction < 0.5:
        raise ValueError("Batch size must be >=2 and validation fraction in (0, 0.5).")
    prices = load_prices(prices_path)
    x, target = make_samples(prices, settings.lookback)
    raw = target.to_numpy(dtype=np.float32, copy=True)
    rows, fold_records = [], []
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)
    for year in range(settings.first_test_year, target.index[-1].year + 1, settings.step_years):
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + settings.step_years, 1, 1)
        past = np.flatnonzero(target.index < start)
        test = np.flatnonzero((target.index >= start) & (target.index < end))
        split = int(len(past) * (1 - settings.validation_fraction))
        train, validation = past[:split], past[split:]
        if len(train) < 2 or len(validation) < 2 or not len(test):
            continue
        mean, scale = fit_scaler(x[train])
        normalized = torch.tensor((x - mean) / scale, dtype=torch.float32)
        y = torch.from_numpy(raw)
        model = PortfolioLSTM(8, settings.hidden_size, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        dataset = TensorDataset(normalized[train], y[train])
        # Avoid a singleton final minibatch, for which sample standard deviation is undefined.
        loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True,
                            drop_last=len(dataset) % settings.batch_size == 1)
        best, state, stale, history = float("inf"), None, 0, []
        for epoch in range(settings.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = sharpe_ratio_loss(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                value = float(sharpe_ratio_loss(model(normalized[validation]), y[validation]))
            if not np.isfinite(value):
                raise RuntimeError("Non-finite validation objective.")
            scheduler.step(value)
            history.append({"epoch": epoch + 1, "validation_loss": value})
            if value < best:
                best, state, stale = value, deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= settings.patience:
                    break
        assert state is not None
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            allocations = torch.cat([model(normalized[chunk]) for chunk in
                                     np.array_split(test, max(1, int(np.ceil(len(test) / 256))))]).numpy()
        for index, weights in zip(test, allocations):
            # Both rolling baselines see returns strictly before this target date.
            past_returns = raw[index - settings.lookback:index]
            if len(past_returns) < settings.lookback:
                # Include pre-sequence history for early target observations.
                past_returns = prices.pct_change(fill_method=None).loc[
                    prices.index < target.index[index]].tail(settings.lookback).to_numpy()
            strategies = {
                "lstm": weights, "equal_weight": np.full(4, 0.25),
                "fixed_40_40_10_10": np.array([0.4, 0.4, 0.1, 0.1]),
                "minimum_variance": optimized_weights(past_returns, "minimum_variance"),
                "maximum_diversification": optimized_weights(past_returns, "maximum_diversification"),
            }
            for strategy, allocation in strategies.items():
                row = {"date": target.index[index], "fold": year, "strategy": strategy,
                       "return": float(allocation @ raw[index])}
                row.update({asset: float(weight) for asset, weight in zip(ASSETS, allocation)})
                rows.append(row)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "mean": torch.from_numpy(mean),
                    "scale": torch.from_numpy(scale), "settings": asdict(settings)},
                   output / f"fold_{year}.pt")
        fold_records.append({"test_start": str(start.date()), "test_end_exclusive": str(end.date()),
                             "train_last": str(target.index[train[-1]].date()),
                             "validation_first": str(target.index[validation[0]].date()),
                             "validation_last": str(target.index[validation[-1]].date()),
                             "train_count": len(train), "validation_count": len(validation),
                             "test_count": len(test), "history": history})
    if not rows:
        raise ValueError("No eligible folds: lower first-test-year or supply more price history.")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "predictions.csv", index=False)
    summary = {name: performance(group["return"].to_numpy()) for name, group in frame.groupby("strategy")}
    metadata = {"settings": asdict(settings), "assets": ASSETS,
                "input_sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
                "torch_version": str(torch.__version__), "numpy_version": np.__version__,
                "pandas_version": pd.__version__, "folds": fold_records,
                "evaluation": "unscaled, zero costs, zero risk-free rate; VIX is a nontradable index proxy",
                "metrics": summary}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
