# Replication scope and reproducibility

## Paper identification

**Zihao Zhang, Stefan Zohren, and Stephen Roberts, “Deep Learning for Portfolio Optimization” (2020).**

- [Canonical arXiv record](https://arxiv.org/abs/2005.13665)
- [Full text, version 3](https://arxiv.org/html/2005.13665v3)
- [Journal DOI](https://doi.org/10.3905/jfds.2020.1.042)
- Affiliation in the paper: **Oxford-Man Institute of Quantitative Finance, University of Oxford**.

Neither source repository named the paper. Attribution is grounded in a distinctive match: the exact four indices, price/return features, 50-day history, single-layer LSTM with a softmax portfolio, direct Sharpe-ratio optimization, two-year expanding retraining, and 25/25/25/25 and 40/40/10/10 allocations. The project author confirmed this is the replicated paper on 2026-11-11. This confirmation does not imply every implementation choice follows the paper.

## Paper vs. course implementation vs. refactor

| Component | Paper | Original course code | Supported refactor |
|---|---|---|---|
| Series | VTI, AGG, DBC, VIX | Same symbols; Yahoo ^VIX | Same snapshot series, explicit nontradable VIX warning |
| Inputs | 50 days of prices + returns, 8 features | Same | Same; configurable lookback |
| LSTM hidden units | 64 | 128 | 128 default; configurable |
| Objective | Direct portfolio Sharpe | Batch Sharpe on globally normalized asset returns | Batch Sharpe on raw next-day asset returns |
| Retraining | Every two years, expanding history | Seven expanding windows | Configurable two-year expanding folds |
| Validation | 10% of training data; test held out | Test fold used for early stopping | Final 10% of past samples; test untouched |
| Checkpoints | Test evaluated after training | Root variants save incompatible filenames; results version only saves final fold model | Best validation state restored and saved per fold |
| Feature normalization | Exact original implementation not available here | Fitted on all windows including future periods | Fitted on training windows only |
| Data period | 2006–2020; test 2011–April 2020 | Snapshots/labels include 2010–2024 | Actual snapshot dates; first test year configurable |
| Risk adjustment | Lagged exponentially weighted asset volatility; target 10% | LSTM rescales by rolling asset volatility then renormalizes weights; baselines differ | No volatility adjustment |
| Costs | Several transaction-cost rates | Omitted | Omitted, stated explicitly |
| MV baseline | Rolling mean–variance/Sharpe selection | Full-period minimum variance; return clipping in results version | Rolling, past-only minimum variance |
| MD baseline | Maximum diversification | Mean minus 0.1×variance, with VIX cap | Genuine weighted-asset-volatility / portfolio-volatility ratio |
| Returns and drawdowns | Annualized metrics and cumulative analysis | Mixed additive cumulative conventions and shifted/generated dates | Actual target dates; compounded wealth; initial wealth included in drawdown |

The refactor changes experiment semantics to remove specific observed defects. Its results must not be pooled with or substituted for archived results. No claim is made that these tests establish strategy profitability.

## Findings in the original scripts

The historical workflows are preserved unchanged in [the code archive](../legacy/code-repository/) and [the results archive](../legacy/results-repository/).

1. **Data leakage:** feature and target-return normalization use full-period moments. The test fold also controls early stopping. The reported comparisons are therefore not established as untouched out-of-sample results.
2. **Final-model evaluation:** the results training script saves only the last model; evaluation predicts across all inputs. Its loader names a different checkpoint from the trainer’s output. That model is absent from the repository.
3. **Missing artifacts:** no trained checkpoints, preprocessed tensors, LSTM daily returns, LSTM weights, or LSTM metric array were committed. Several plotting scripts reference these missing files. Do not fabricate or reconstruct missing series from PNGs.
4. **Calendar alignment:** some scripts create generic business-day dates, then shift LSTM dates by six months. Actual date/return mapping cannot be recovered confidently from the figures alone.
5. **Volatility adjustment:** the LSTM renormalizes inverse-volatility-adjusted weights back to sum one. The common target multiplier cancels, so this is not the paper’s exposure-level 10% volatility targeting. Other baselines use portfolio-level adjustments, sometimes including contemporaneous return information.
6. **Baseline labels:** the MD script’s objective is expected return minus variance, not diversification ratio. The comparison script calls its final column “Mean Variance” while an original figure labels it “Maximum Diversity.” The refactor uses an explicitly historical label rather than silently equating these strategies.
7. **Cumulative conventions:** some outputs use `cumsum`, others use `1 + cumsum`, and others compound. Original plot labels are not proof of consistent compounded wealth.
8. **Asset order:** snapshot columns are AGG, DBC, VTI, VIX; some original fixed-weight interpretations assume a different order. New code selects VTI, AGG, DBC, VIX by name.

## What the historical table establishes

The recorded values come from [the archived comparison script](../legacy/results-repository/tables_comparison.py). Reporting parses its literal numeric values without executing the script. The equal/fixed and MV metric arrays agree with the corresponding table values at their reported precision; the complete daily series needed to verify every strategy are absent. The LSTM and historically MD-labeled table entries remain reported summaries.

Historical Sortino uses the standard deviation of negative returns only. The refactor uses full-sample downside deviation relative to zero. Annualized return is arithmetic mean ×252, **not CAGR**, in both contexts.

## Validation performed during consolidation

- All 58 substantive source files are preserved byte-for-byte in the curated checkout; Git-blob fingerprints also record Windows line-ending differences.
- All 14 source figures remain unchanged.
- Numerical/unit tests and a synthetic two-fold training/evaluation smoke test pass.
- A one-epoch snapshot-based smoke run validates the CLI workflow; it does not validate the historical performance figures.
- No large performance experiment or data download is required to inspect the project.

## Extensions for an investable evaluation

Use a specified tradable volatility product (and its own history), include transaction costs and turnover, implement lagged exposure-level risk scaling, fix price-adjustment/data-provider versions, and run repeated seeded out-of-sample experiments. These are future experiments, not already-completed findings.