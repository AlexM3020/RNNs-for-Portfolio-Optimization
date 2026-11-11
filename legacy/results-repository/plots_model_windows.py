import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

plt.style.use('seaborn-v0_8-muted')

# --------------------------
# Load saved data
# --------------------------
portfolio_returns = np.load("portfolio_returns_windows.npy")
log_cum_returns = np.load("log_cum_returns_windows.npy")
drawdown = np.load("drawdown_windows.npy")
predicted_weights = np.load("predicted_weights_windows.npy")
dates = pd.to_datetime(np.load("dates_windows.npy")) + pd.DateOffset(months=6)

log_returns = np.load("log_returns_windows.npy")
# --------------------------
# Plot 1 – Cumulative Return (with optional log scale)
# --------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, 1+ np.cumsum(portfolio_returns), label='Cumulative Yield', color='green')
# ax.set_yscale('log')  # Uncomment for log-scale compounding
ax.set_title('Cumulative Portfolio Return (2010–2024)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


# --------------------------
# Plot 2 – Drawdown Smoothed
# --------------------------
drawdown_series = pd.Series(drawdown, index=dates).rolling(window=10, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(drawdown_series.index, drawdown_series, label='Drawdown (10-day MA)', color='red')
ax.set_title('📉 Daily Drawdown (Peak-to-Trough Loss)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# --------------------------
# Plot 3 – Daily Returns
# --------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates, portfolio_returns, label='Daily Returns', color='blue', linewidth=0.8)
ax.set_title('Daily Returns', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Return')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# --------------------------
# Histogram of Returns
# --------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(portfolio_returns, bins=50, color='steelblue', edgecolor='black')
ax.set_title('Distribution of Daily Returns', fontsize=13)
ax.set_xlabel('Return')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# --------------------------
# Plot 4A – Weights Moving Average
# --------------------------
etf_names = ["VTI", "AGG", "DBC", "VIX"]
weights_df = pd.DataFrame(predicted_weights, index=dates)
weights_smooth = weights_df.rolling(window=30, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(weights_smooth.shape[1]):
    ax.plot(weights_smooth.index, weights_smooth.iloc[:, i], label=etf_names[i])
ax.set_title('Smoothed Evolution of Portfolio Weights (30-day MA)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Weight')
ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# --------------------------
# Plot 4B – Stackplot of Weights
# --------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(dates, predicted_weights.T, labels=etf_names)
ax.set_title('Portfolio Weights (Stacked Area Plot)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Weight')
ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='upper left')
ax.grid(True)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# --------------------------
# Load additional saved data
# --------------------------
log_returns = np.load("log_returns_windows.npy")
annualized_vol = np.load("annualized_vol_windows.npy")
vol_adjustments = np.load("vol_adjustments_windows.npy")
metrics = np.load("metrics_windows.npy")
sharpe_ratio, sortino_ratio, max_drawdown, expected_return, annual_std, positive_pct, avg_p_to_l = metrics

# --------------------------
# Plot 5 – Rolling Annualized Volatility
# --------------------------
fig, ax = plt.subplots(figsize=(12, 4))
vol_series = pd.Series(vol_adjustments.mean(axis=1), index=dates)
ax.plot(vol_series.rolling(30).mean(), label='30-day MA', color='purple')
ax.axhline(y=1, color='red', linestyle='--', label='Target Vol (10%)')
ax.set_title('Portfolio Annualized Volatility (30-day MA)', fontsize=14)
ax.set_ylabel('Volatility')
ax.legend()
plt.tight_layout()
plt.show()

# --------------------------
# Plot 6 – Rolling Sharpe Ratio
# --------------------------
rolling_sharpe = (pd.Series(portfolio_returns).rolling(252).mean() /
                  pd.Series(portfolio_returns).rolling(252).std()) * np.sqrt(252)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates[-len(rolling_sharpe):], rolling_sharpe, color='green')
ax.axhline(y=sharpe_ratio, color='gray', linestyle='--', label=f'Overall Sharpe: {sharpe_ratio:.2f}')
ax.set_title('1-Year Rolling Sharpe Ratio', fontsize=14)
ax.set_ylabel('Sharpe Ratio')
ax.legend()
plt.tight_layout()
plt.show()

# --------------------------
# Plot 6B – Rolling Sortino Ratio
# --------------------------
rolling_downside = pd.Series(portfolio_returns).rolling(252).apply(
    lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 0 else np.nan)
rolling_sortino = pd.Series(portfolio_returns).rolling(252).mean() / rolling_downside * np.sqrt(252)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates[-len(rolling_sortino):], rolling_sortino, color='orange')
ax.axhline(y=sortino_ratio, color='gray', linestyle='--', label=f'Overall Sortino: {sortino_ratio:.2f}')
ax.set_title('1-Year Rolling Sortino Ratio', fontsize=14)
ax.set_ylabel('Sortino Ratio')
ax.legend()
plt.tight_layout()
plt.show()



