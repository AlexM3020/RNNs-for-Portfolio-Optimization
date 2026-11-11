import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --------------------------
# Load Deep Learning strategy data
# --------------------------
portfolio_returns_dl = np.load("portfolio_returns_windows.npy")
cum_returns_dl = np.load("cum_returns_windows.npy")
log_returns_dl = np.load("log_returns_windows.npy")
dates = pd.to_datetime(np.load("dates_windows.npy")) + pd.DateOffset(months=6)
drawdown_dl = np.load("drawdown_windows.npy")

# --------------------------
# Load MD strategy data
# --------------------------
cum_returns_md = pd.read_csv("md_strategy_cumulative_returns_scaled_dynamic.csv", index_col=0, parse_dates=True).squeeze()
portfolio_returns_md = pd.read_csv("portfolio_returns_md_dynamic.csv", index_col=0, parse_dates=True)
log_returns_md = np.log1p(portfolio_returns_md)
drawdown_md = np.load("md_drawdown.npy")

# Allineamento date MD
cum_returns_md = cum_returns_md.loc[(cum_returns_md.index >= dates[0]) & (cum_returns_md.index <= dates[-1])]
portfolio_returns_md = portfolio_returns_md.loc[cum_returns_md.index]
log_returns_md = log_returns_md.loc[cum_returns_md.index]
drawdown_md = drawdown_md[:len(cum_returns_md)]
drawdown_md = pd.Series(drawdown_md, index=cum_returns_md.index)

# --------------------------
# Load MV strategy data
# --------------------------
portfolio_returns_mv = np.load("mv_portfolio_returns.npy")
cum_returns_mv = np.load("mv_cum_returns.npy")
drawdown_mv = np.load("mv_drawdown.npy")
cum_returns_mv = 1+ cum_returns_mv
# --------------------------
# Load Allocation 1
# --------------------------
portfolio_returns_alloc1 = np.load("portfolio_returns_fixed.npy")
cum_returns_alloc1 = np.load("cum_returns_fixed.npy")
drawdown_alloc1 = np.load("drawdown_fixed.npy")
cum_returns_alloc1  = 1+ cum_returns_alloc1
# --------------------------
# Load Allocation 2
# --------------------------
portfolio_returns_alloc2 = np.load("portfolio_returns_fixed_40.npy")
cum_returns_alloc2 = np.load("cum_returns_fixed_40.npy")
drawdown_alloc2 = np.load("drawdown_fixed_40.npy")
cum_returns_alloc2 = 1+ cum_returns_alloc2
# --------------------------
# Allineamento e conversione
# --------------------------
dates_pd = pd.to_datetime(dates)

cum_returns_dl = pd.Series(cum_returns_dl, index=dates_pd)
portfolio_returns_dl = pd.Series(portfolio_returns_dl, index=dates_pd)
drawdown_dl = pd.Series(drawdown_dl, index=dates_pd)

len_common = len(dates_pd)

def trim_and_series(arr):
    return pd.Series(arr[:len_common], index=dates_pd)

cum_returns_mv = trim_and_series(cum_returns_mv)
drawdown_mv = trim_and_series(drawdown_mv)
portfolio_returns_mv = trim_and_series(portfolio_returns_mv)

cum_returns_alloc1 = trim_and_series(cum_returns_alloc1)
drawdown_alloc1 = trim_and_series(drawdown_alloc1)
portfolio_returns_alloc1 = trim_and_series(portfolio_returns_alloc1)

cum_returns_alloc2 = trim_and_series(cum_returns_alloc2)
drawdown_alloc2 = trim_and_series(drawdown_alloc2)
portfolio_returns_alloc2 = trim_and_series(portfolio_returns_alloc2)

# --------------------------
# Plot 1 – Cumulative Returns (Sovrapposti)
# --------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(cum_returns_dl, label='Deep Learning', color='red', linewidth=1)
ax.plot(cum_returns_md, label='Mean Variance', color='blue', linewidth=1)
ax.plot(cum_returns_mv, label='Minimum Variance', color='green', linewidth=1)
ax.plot(cum_returns_alloc1, label='Allocation 1 (Equal)', color='orange', linewidth=1)
ax.plot(cum_returns_alloc2, label='Allocation 2 [40, 40, 10, 10]', color='purple', linewidth=1)

ax.set_title("Cumulative Portfolio Return (Volatility Scaled)")
ax.set_ylabel("Portfolio Value")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()

# --------------------------
# Plot 2 – Daily Returns (subplot impilati)
# --------------------------
strategies_returns = {
    'Deep Learning': portfolio_returns_dl,
    'Maximum Diversity': portfolio_returns_md.squeeze(),
    'Minimum Variance': portfolio_returns_mv,
    'Allocation 1': portfolio_returns_alloc1,
    'Allocation 2': portfolio_returns_alloc2
}

colors = ['red', 'blue', 'green', 'orange', 'purple']

fig, axes = plt.subplots(len(strategies_returns), 1, figsize=(12, 10), sharex=True)

for ax, (name, series), color in zip(axes, strategies_returns.items(), colors):
    ax.plot(series.rolling(5).mean(), label=f'{name} (5d MA)', color=color, linewidth=1)
    ax.set_title(f"Daily Returns – {name}")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()

# --------------------------
# Plot 3 – Drawdown (subplot impilati)
# --------------------------
strategies_drawdowns = {
    'Deep Learning': drawdown_dl,
    'Maximum Diversity': drawdown_md,
    'Minimum Variance': drawdown_mv,
    'Allocation 1': drawdown_alloc1,
    'Allocation 2': drawdown_alloc2
}

fig, axes = plt.subplots(len(strategies_drawdowns), 1, figsize=(12, 10), sharex=True)

for ax, (name, series), color in zip(axes, strategies_drawdowns.items(), colors):
    ax.plot(series, label=name, color=color, linewidth=1)
    ax.set_title(f"Drawdown – {name}")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')

axes[-1].set_xlabel("Date")
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()