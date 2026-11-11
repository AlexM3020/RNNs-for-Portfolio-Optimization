import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# === Load MV outputs ===
portfolio_returns = np.load("mv_portfolio_returns.npy")
cum_returns = np.load("mv_cum_returns.npy")
drawdown = np.load("mv_drawdown.npy")
predicted_weights = np.load("mv_weights.npy")
dates = np.load("mv_dates.npy")

etf_names = ["VTI", "AGG", "DBC", "VIX"]  # update if more assets exist

# === Plot 1: Cumulative Returns ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, cum_returns, label='Cumulative Return (MV)', color='green')
ax.set_title('MV Portfolio – Cumulative Return (2010–2024)')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# === Plot 2: Drawdown (Smoothed) ===
drawdown_series = pd.Series(drawdown, index=dates)
drawdown_smoothed = drawdown_series.rolling(window=10, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(drawdown_smoothed.index, drawdown_smoothed.values, label='Drawdown (10d MA)', color='red')
ax.set_title('MV Portfolio – Drawdown (Smoothed)')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# === Plot 3: Daily Returns ===
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates, portfolio_returns, label='Daily Returns (MV)', color='blue', linewidth=0.8)
ax.set_title('MV Portfolio – Daily Returns')
ax.set_xlabel('Date')
ax.set_ylabel('Return')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# === Plot 4A: Smoothed Weights (Line Plot) ===
weights_df = pd.DataFrame(predicted_weights, index=dates)
weights_smooth = weights_df.rolling(window=30, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(weights_smooth.shape[1]):
    label = etf_names[i] if i < len(etf_names) else f"Asset {i+1}"
    ax.plot(weights_smooth.index, weights_smooth.iloc[:, i], label=label)
ax.set_title('MV Portfolio – Smoothed Weights (30-day MA)')
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

# === Plot 4B: Stackplot of Weights ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(dates, predicted_weights.T, labels=etf_names[:predicted_weights.shape[1]])
ax.set_title('MV Portfolio – Asset Weights (Stacked)')
ax.set_xlabel('Date')
ax.set_ylabel('Weight')
ax.set_ylim(0, 1)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend(loc='upper left')
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
