# Historical experiment archive

Original artifacts are preserved unchanged, including duplicate exports and earlier variants. This prevents a presentation cleanup from destroying potentially useful evidence. Supported code does not automatically read these artifacts as a validated backtest.

## Curated summary

- [Recorded metric CSV](summary/recorded_metrics.csv)
- [New summary chart](summary/recorded_metrics.png)
- Source: [archived table-generation script](../../legacy/results-repository/tables_comparison.py)

Regenerate with `portfolio-lstm report --output runs/historical-report` from the repository root. The chart visualizes recorded metrics, not newly measured returns.

## Original figure gallery

| Figure | Content visible in the original image |
|---|---|
| [1](figures/Figure_1.png) | LSTM metric summary |
| [2](figures/Figure_2.png) | Five-strategy recorded comparison table |
| [3](figures/Figure_3.png) | LSTM log cumulative analysis and drawdown |
| [4](figures/Figure_4.png) | LSTM crisis-period detail |
| [5](figures/Figure_5.png) | Historical cumulative-return curve |
| [6](figures/Figure_6.png) | Daily returns |
| [7](figures/Figure_7.png) | Daily-return distribution |
| [8](figures/Figure_8.png) | Smoothed allocation evolution |
| [9](figures/Figure_9.png) | Stacked allocation weights |
| [10](figures/Figure_10.png) | Rolling Sharpe |
| [11](figures/Figure_11.png) | Rolling Sortino |
| [12](figures/Figure_12.png) | Strategy cumulative comparison |
| [13](figures/Figure_13.png) | Strategy daily-return panels |
| [14](figures/Figure_14.png) | Strategy drawdown panels |

Original cumulative/date conventions are inconsistent across scripts. No image has been redated, relabeled, or redrawn to imply validation of unavailable daily series.

## Numeric artifact inventory

[artifacts](artifacts/) contains 11 NumPy arrays and 4 CSV files:

- `metrics_fixed.npy` and `metrics_fixed_uniform.npy`: identical 7-element equal-allocation metrics.
- `metrics_fixed_40.npy` and `metrics_fixed_uniform_40.npy`: identical 7-element second-allocation metrics.
- `mv_metrics.npy`: 7 metrics; `mv_cum_returns.npy` and `mv_dates.npy`: 3,721 observations, with generated dates.
- `md_portfolio_returns.npy` and the oddly named `md_portfolio_returns.csv.npy`: identical 3,773-element arrays; do not assume they are the final dynamically scaled run.
- `md_drawdown.npy`: 3,673 observations; this length differs from the other MD arrays.
- Three cumulative-return CSV variants and one scaled-return CSV are kept with their original names.

Metric arrays use this order: **Sharpe, Sortino, maximum drawdown, annualized mean return, annualized volatility, percentage positive, average gain/average loss**. The complete original LSTM daily arrays and model checkpoints are absent. [Replication notes](../../docs/replication.md) explain the resulting limits.

All original filenames remain in the archive for traceability; machine-readable [migration fingerprints](../../docs/migration_manifest.json) identify the source commit and destination of every file.