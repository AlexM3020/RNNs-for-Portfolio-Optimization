import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

plt.style.use('seaborn-v0_8-muted')  # Optional for cleaner theme

# --------------------------
# Load saved data
# --------------------------
portfolio_returns = np.load("portfolio_returns_windows.npy")
cum_returns = np.load("log_cum_returns_windows.npy")
drawdown = np.load("drawdown_windows.npy")
predicted_weights = np.load("predicted_weights_windows.npy")
dates = np.load("dates_windows.npy")
etf_names = ["VTI", "AGG", "DBC", "VIX"]

# --------------------------
# Plot 8 – Cumulative Return per ETF
# --------------------------
cumulative_returns = (predicted_weights * np.array(predicted_weights.shape[0] * [portfolio_returns]).T).cumsum(axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(cumulative_returns.shape[1]):
    ax.plot(dates, cumulative_returns[:, i], label=etf_names[i])
ax.set_title('Cumulative Return per ETF (Weighted Contribution)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# --------------------------
# Enhanced Drawdown Plot
# --------------------------
fig, ax = plt.subplots(figsize=(14, 5))

# Convert drawdown to percentage and smooth with 21-day EMA (1 month)
drawdown_pct = pd.Series(drawdown * 100, index=dates).ewm(span=21).mean()

# Fill under curve for better visibility
ax.fill_between(dates, drawdown_pct, 0, color='salmon', alpha=0.3)

# Main drawdown line
ax.plot(dates, drawdown_pct, color='darkred', linewidth=1.5, label='Drawdown (21-day EMA)')

# Highlight max drawdown
max_dd_idx = np.argmin(drawdown)
max_dd_date = dates[max_dd_idx]
max_dd_value = drawdown_pct[max_dd_idx]
ax.scatter(max_dd_date, max_dd_value, color='black', s=100,
           label=f'Max Drawdown: {max_dd_value:.1f}%', zorder=5)

# Add horizontal grid lines
ax.yaxis.grid(True, linestyle=':', alpha=0.7)

# Formatting
ax.set_title('Portfolio Drawdown Analysis (2010-2024)', fontsize=14, pad=20)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Drawdown (%)', fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(loc='lower left', framealpha=1)

# Add crisis period annotations
crisis_periods = [
    ('2011-08-01', '2011-09-30', 'US Debt Crisis'),
    ('2020-02-20', '2020-04-30', 'COVID-19 Crash')
]

for start, end, label in crisis_periods:
    mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
    if any(mask):
        crisis_dd = drawdown_pct[mask]
        ax.plot(dates[mask], crisis_dd, color='black', linewidth=2.5)
        ax.text(pd.to_datetime(end), crisis_dd.min()-2, label,
                ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()