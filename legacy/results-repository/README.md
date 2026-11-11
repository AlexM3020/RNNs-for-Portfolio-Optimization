# Portfolio Optimization via LSTM

Course project for *Introduction to Control and Machine Learning* (ICM).

An LSTM is trained to dynamically allocate a portfolio across four asset classes,
directly optimizing for risk-adjusted return, and is benchmarked against classic
portfolio-construction baselines on 2010-2024 daily data.

## Assets

| Ticker | Asset class |
|---|---|
| VTI | US equities |
| AGG | US bonds |
| DBC | Commodities |
| ^VIX | Volatility index |

## Approach

**Data.** Daily close prices for all four tickers are pulled from Yahoo Finance
(2010-2024). Daily returns are computed and combined with prices into a single
feature set.

**Preprocessing.** Features are cut into sliding 50-day windows (z-score
normalized), so each training sample is "the last 50 days" and the model
learns from short-term historical context rather than a single snapshot.

**Model.** A single-layer LSTM (`model.py`) consumes a 50-day window and maps
its final hidden state through a linear layer + softmax into 4 portfolio
weights that sum to 1.

**Loss.** Instead of predicting returns and allocating from the prediction,
the model is trained directly on portfolio performance: `sharp_ratio.py`
implements a differentiable Sharpe ratio loss, computed batch-wise on the
weighted portfolio returns, so the network learns to directly maximize
risk-adjusted return rather than raw accuracy.

**Training.** `training_windows.py` trains with walk-forward validation:
seven rolling train/test splits (e.g. train on 2010-2018, test on 2018-2020,
then roll forward) rather than one static train/test split, with early
stopping and LR scheduling driven by rolling validation Sharpe.

**Evaluation.** `model_eval.py` applies inverse-volatility scaling on top of
the model's raw weights (10% annualized target volatility) before computing
final performance metrics, so every strategy below is compared on a
volatility-scaled, apples-to-apples basis.

**Baselines.** The LSTM is compared against three classic strategies, each
volatility-scaled the same way: Minimum Variance (`mv.py`), Maximum
Diversification (`md.py`), and two static fixed allocations (`fixed_alloc.py`).

## Results (2010-2024, volatility-scaled)

| Strategy | Ann. Return | Sharpe | Sortino | Max Drawdown |
|---|---|---|---|---|
| **LSTM** | **16.4%** | **1.66** | 2.23 | -14.9% |
| Minimum Variance | 15.1% | 1.48 | **2.31** | -22.9% |
| Fixed Allocation 2 (40/40/10/10) | 10.7% | 1.06 | 1.93 | -11.9% |
| Fixed Allocation 1 (equal) | 9.7% | 0.95 | 1.91 | **-7.6%** |
| Maximum Diversification | 3.2% | 0.31 | 0.53 | -23.1% |

The LSTM strategy wins on return and Sharpe, but it's not a clean sweep:
Minimum Variance still has the better Sortino ratio, and both fixed
allocations have shallower drawdowns. Full comparison plots (cumulative
return, daily return, and drawdown, all side by side) are in
`strategy_comparison_full.png`.

## Repository structure

```
data.py                   # Downloads prices, computes returns/features
preprocessing.py          # Sliding windows + normalization
model.py                  # LSTM architecture
sharp_ratio.py            # Differentiable Sharpe ratio loss
training_windows.py       # Walk-forward rolling-window training loop
model_eval.py             # Volatility scaling + performance metrics for the LSTM
mv.py                     # Minimum Variance baseline
md.py                     # Maximum Diversification baseline
fixed_alloc.py             # Static fixed-allocation baselines
compare.py                # Overlaid comparison plots (returns, drawdown)
tables_comparison.py      # Final metrics comparison table
drawdown_analysis.py      # Drawdown-focused analysis
images/                   # Saved figures
```



## How to run

```bash
pip install torch pandas numpy scikit-learn yfinance matplotlib scipy

python data.py              # Download & prepare price/return data
python preprocessing.py     # Build normalized sliding-window tensors
python training_windows.py  # Train the LSTM with walk-forward validation
python model_eval.py        # Evaluate the trained model
python mv.py                # Run Minimum Variance baseline
python md.py                # Run Maximum Diversification baseline
python fixed_alloc.py       # Run fixed-allocation baselines
python compare.py           # Generate comparison plots
python tables_comparison.py # Generate the final metrics table
```

