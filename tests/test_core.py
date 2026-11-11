import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from portfolio_lstm.baselines import optimized_weights
from portfolio_lstm.data import ASSETS, fit_scaler, load_prices, make_samples
from portfolio_lstm.metrics import performance, wealth_and_drawdown
from portfolio_lstm.model import PortfolioLSTM, sharpe_ratio_loss

ROOT = Path(__file__).resolve().parents[1]


def prices(n=80):
    rng = np.random.default_rng(1)
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.001, 0.01, (n, 4)), axis=0),
                        index=pd.bdate_range("2010-01-01", periods=n), columns=ASSETS)


def test_simplex_and_gradient():
    torch.set_num_threads(2)
    model = PortfolioLSTM(hidden_size=8)
    weights = model(torch.randn(10, 5, 8))
    assert weights.shape == (10, 4)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(1), torch.ones(10))
    loss = sharpe_ratio_loss(weights, torch.randn(10, 4) * 0.01)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_loss_rejects_invalid_input():
    with pytest.raises(ValueError):
        sharpe_ratio_loss(torch.ones(1, 4), torch.ones(1, 4))
    with pytest.raises(ValueError):
        sharpe_ratio_loss(torch.ones(3, 4), torch.full((3, 4), float("nan")))


def test_next_day_alignment():
    frame = prices()
    x, y = make_samples(frame, 5)
    assert len(x) == len(frame) - 6
    assert y.index[0] == frame.index[6]
    np.testing.assert_allclose(x[0, -1, :4], frame.iloc[5], rtol=1e-6)
    np.testing.assert_allclose(y.iloc[0], frame.iloc[6] / frame.iloc[5] - 1)
    np.testing.assert_allclose(x[0, -1, 4:], frame.iloc[5] / frame.iloc[4] - 1, rtol=1e-6)


def test_training_only_scaler():
    x, _ = make_samples(prices(), 5)
    train = x[:10].copy()
    mean, scale = fit_scaler(train)
    x[10:] = 1e9
    mean2, scale2 = fit_scaler(x[:10])
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_allclose(scale, scale2, rtol=1e-14, atol=0)
    _, constant_scale = fit_scaler(np.ones((3, 5, 8)))
    np.testing.assert_array_equal(constant_scale, np.ones(8))


def test_compounding_and_initial_drawdown():
    wealth, dd = wealth_and_drawdown(np.array([-0.1, 0.1]))
    np.testing.assert_allclose(wealth, [0.9, 0.99])
    np.testing.assert_allclose(dd, [-0.1, -0.01])
    assert performance(np.zeros(3))["sharpe"] is None
    with pytest.raises(ValueError):
        wealth_and_drawdown(np.array([-1.0]))


@pytest.mark.parametrize("strategy", ["minimum_variance", "maximum_diversification"])
def test_baseline_simplex(strategy):
    history = np.random.default_rng(42).normal(0, [0.01, 0.02, 0.015, 0.03], (100, 4))
    weights = optimized_weights(history, strategy)
    assert (weights >= 0).all()
    assert abs(weights.sum() - 1) < 1e-8


def test_snapshot_asset_order():
    frame = load_prices(ROOT / "data/snapshots/etf_data_2010_2024.csv")
    assert tuple(frame.columns) == ASSETS
    assert frame.index.is_monotonic_increasing


def test_reject_unsorted_prices(tmp_path):
    path = tmp_path / "prices.csv"
    prices().iloc[::-1].to_csv(path)
    with pytest.raises(ValueError, match="increasing"):
        load_prices(path)


def test_archive_table_not_invented():
    from portfolio_lstm.reporting import historical_table
    table = historical_table(ROOT)
    assert table.loc["LSTM", "Sharpe Ratio"] == 1.6607
    assert table.loc["LSTM", "Expected Return (Annualized)"] == 0.1641
    assert len(table) == 5


@pytest.mark.parametrize("filename,strategy", [
    ("metrics_fixed.npy", "Fixed Allocation 1"),
    ("metrics_fixed_40.npy", "Fixed Allocation 2"),
    ("mv_metrics.npy", "Minimum Variance"),
])
def test_saved_metrics_agree_with_reported_precision(filename, strategy):
    from portfolio_lstm.reporting import historical_table
    values = np.load(ROOT / "results/historical/artifacts" / filename, allow_pickle=False)
    table = historical_table(ROOT)
    columns = ["Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown", "Expected Return (Annualized)",
               "Volatility (Annualized)", "% Positive Returns", "Avg Gain/Avg Loss"]
    for i, (value, column) in enumerate(zip(values, columns)):
        assert round(float(value), 2 if i == 5 else 4) == table.loc[strategy, column]


def test_documentation_links_exist():
    import re
    documents = [ROOT / "README.md", *ROOT.glob("docs/*.md"),
                 ROOT / "data/README.md", ROOT / "results/historical/README.md"]
    for document in documents:
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if not target.startswith(("https:", "http:", "#")):
                assert (document.parent / target.split("#")[0]).exists(), (document, target)


def test_imports_have_no_io_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ["model", "data", "metrics", "baselines", "backtest", "reporting", "cli"]:
        importlib.reload(importlib.import_module("portfolio_lstm." + name))
    assert not list(tmp_path.iterdir())


def test_walk_forward_smoke(tmp_path):
    from portfolio_lstm.backtest import Settings, run_backtest
    path, output = tmp_path / "prices.csv", tmp_path / "run"
    frame = prices(560)
    frame.to_csv(path)
    settings = Settings(lookback=5, hidden_size=4, epochs=1, batch_size=32,
                        first_test_year=2011, step_years=1)
    run_backtest(path, output, settings)
    metadata = json.loads((output / "summary.json").read_text())
    predictions = pd.read_csv(output / "predictions.csv", parse_dates=["date"])
    assert len(metadata["folds"]) == 2
    for fold in metadata["folds"]:
        assert fold["train_last"] < fold["validation_first"] <= fold["validation_last"] < fold["test_start"]
    assert not predictions.duplicated(["date", "strategy"]).any()
    assert predictions.groupby("strategy").size().nunique() == 1
    np.testing.assert_allclose(predictions[list(ASSETS)].sum(axis=1), 1, atol=1e-6)
    actual = frame.pct_change().loc[predictions.loc[predictions.strategy == "equal_weight", "date"]].mean(axis=1)
    np.testing.assert_allclose(predictions.loc[predictions.strategy == "equal_weight", "return"], actual, atol=1e-8)
    with pytest.raises(ValueError, match="never overwritten"):
        run_backtest(path, output, settings)


def test_future_prices_cannot_change_selected_model(tmp_path):
    from portfolio_lstm.backtest import Settings, run_backtest
    settings = Settings(lookback=5, hidden_size=4, epochs=1, batch_size=32, first_test_year=2011)
    frame = prices(280)
    original, perturbed = tmp_path / "original.csv", tmp_path / "perturbed.csv"
    frame.to_csv(original)
    future = frame.index >= pd.Timestamp("2011-01-01")
    frame.loc[future] *= np.linspace(1.1, 1.8, int(future.sum()))[:, None]
    frame.to_csv(perturbed)
    run_backtest(original, tmp_path / "first", settings)
    run_backtest(perturbed, tmp_path / "second", settings)
    first = torch.load(tmp_path / "first/fold_2011.pt", weights_only=True)
    second = torch.load(tmp_path / "second/fold_2011.pt", weights_only=True)
    assert torch.equal(first["mean"], second["mean"])
    assert torch.equal(first["scale"], second["scale"])
    assert all(torch.equal(value, second["state_dict"][key]) for key, value in first["state_dict"].items())
