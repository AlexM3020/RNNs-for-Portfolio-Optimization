import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# --------------------------
# Carica i dati salvati
# --------------------------
portfolio_returns = np.load("portfolio_returns_windows.npy")
cum_returns = np.load("log_cum_returns_windows.npy")
drawdown = np.load("drawdown_windows.npy")
predicted_weights = np.load("predicted_weights_windows.npy")  # shape: (n_days, n_assets)
dates = np.load("dates_windows.npy")  # datetime64[D]

# --------------------------
# Calcola rendimento cumulativo


# --------------------------
# Plot 1 – Rendimento Cumulativo
# --------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates, cum_returns, label='Cumulative Yield', color='green')
ax.set_title('Cumulative Portfolio Return (2010–2024)')
ax.set_xlabel('Data')
ax.set_ylabel('Portfolio Value')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# --------------------------
# Plot 2 – Drawdown (smoothed)
# --------------------------
drawdown_series = pd.Series(drawdown, index=dates)
drawdown_smoothed = drawdown_series.rolling(window=10, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(drawdown_smoothed.index, drawdown_smoothed.values, label='Drawdown (media 10g)', color='red')
ax.set_title('📉 Daily Drawdown (Peak-to-Trough Loss)')
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
# Plot 3 – Ritorni Giornalieri
# --------------------------
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dates, portfolio_returns, label='Daily Returns', color='blue', linewidth=0.8)
ax.set_title('Daily Returns')
ax.set_xlabel('Date')
ax.set_ylabel('Return')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()



etf_names = ["VTI", "AGG", "DBC", "VIX"]
# --------------------------
# Plot 4A – Pesi con media mobile
# --------------------------
weights_df = pd.DataFrame(predicted_weights, index=dates)
weights_smooth = weights_df.rolling(window=30, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(weights_smooth.shape[1]):
    ax.plot(weights_smooth.index, weights_smooth.iloc[:, i], label=etf_names[i])
ax.set_title('Smoothed Evolution of Portfolio Weights (30g Moving Average)')
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
# Plot 4B – Pesi con stackplot (area)
# --------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(dates, predicted_weights.T, labels=etf_names)
ax.set_title('Portfolio Weights (Stacked Area Plot)')
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
