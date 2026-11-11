"""Render the recorded course-project table; never imply it is a fresh backtest."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def historical_table(root: Path) -> pd.DataFrame:
    source = root / "legacy/results-repository/tables_comparison.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "data" for t in node.targets):
            # Parse the archived literal; do not execute plotting or training scripts.
            if not isinstance(node.value, ast.Dict):
                raise ValueError("Expected archived table dictionary.")
            metric_names = None
            for assignment in tree.body:
                if isinstance(assignment, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "metric_map" for t in assignment.targets
                ):
                    metric_names = list(ast.literal_eval(assignment.value))
            if metric_names is None:
                raise ValueError("Archived metric names were not found.")
            recorded = {}
            for key, value in zip(node.value.keys, node.value.values):
                name = ast.literal_eval(key)
                recorded[name] = metric_names if name == "Metric" else ast.literal_eval(value)
            return pd.DataFrame(recorded).set_index("Metric").T.rename(
                index={"Deep Learning": "LSTM", "Mean Variance": "Historical mean–variance (MD label)"})
    raise ValueError("Archived results table was not found.")


def render_historical(root: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = historical_table(root)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "recorded_metrics.csv", index_label="strategy")
    colors = ["#176B9B", "#449B86", "#8794A1", "#C59342", "#786A9C"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    labels = ["LSTM", "Minimum\nvariance", "Equal\nallocation", "40/40/10/10", "Historical\nMD label"]
    metrics = [("Expected Return (Annualized)", "Annualized mean return (%)", 100),
               ("Sharpe Ratio", "Sharpe ratio", 1),
               ("Maximum Drawdown", "Maximum drawdown (%)", 100)]
    for ax, (metric, title, multiplier) in zip(axes, metrics):
        values = table[metric].to_numpy() * multiplier
        ax.bar(np.arange(len(values)), values, color=colors, width=0.65)
        ax.set_xticks(np.arange(len(values)), labels, fontsize=8)
        ax.set_title(title, fontsize=11, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.annotate(f"{value:.2f}", (i, value), xytext=(0, 4 if value >= 0 else -13),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.margins(y=0.2)
    fig.suptitle("Portfolio allocation · recorded course-project results\n"
                 "Historical summary, not an independently verified out-of-sample backtest",
                 fontsize=13, fontweight="bold")
    fig.savefig(output / "recorded_metrics.png", dpi=180, facecolor="white")
    plt.close(fig)


def plot_run(run: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .metrics import wealth_and_drawdown

    frame = pd.read_csv(run / "predictions.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for name, group in frame.groupby("strategy"):
        group = group.sort_values("date")
        wealth, drawdown = wealth_and_drawdown(group["return"].to_numpy())
        axes[0].plot(group["date"], wealth, label=name.replace("_", " "))
        axes[1].plot(group["date"], drawdown * 100)
    axes[0].set_title("Refactored walk-forward run · unscaled, zero transaction costs")
    axes[0].set_ylabel("Compounded wealth (start = 1)")
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(run / "performance.png", dpi=180)
    plt.close(fig)
