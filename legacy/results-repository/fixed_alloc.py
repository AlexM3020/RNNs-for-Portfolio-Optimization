import numpy as np
import pandas as pd

# --------------------------
# Load daily returns
# --------------------------
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
returns = returns.dropna()
dates = returns.index
n_assets = returns.shape[1]

# --------------------------
# Fixed Allocation Strategy
# --------------------------
weights = np.array([0.40, 0.40, 0.10, 0.10])
portfolio_returns = returns @ weights

# --------------------------
# Volatility Scaling
# --------------------------
sigma_target = 0.10
window = 50
rolling_vol = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
vol_adjustment = sigma_target / (rolling_vol + 1e-8)
portfolio_returns_scaled = portfolio_returns * vol_adjustment
portfolio_returns_scaled.fillna(0, inplace=True)

# --------------------------
# Metrics
# --------------------------
mean_daily = np.mean(portfolio_returns_scaled)
std_daily = np.std(portfolio_returns_scaled)
expected_return = mean_daily * 252
volatility = std_daily * np.sqrt(252)
sharpe_ratio = expected_return / (volatility + 1e-8)

negative_returns = portfolio_returns_scaled[portfolio_returns_scaled < 0]
downside_std = np.std(negative_returns)
annual_downside = downside_std * np.sqrt(252)
sortino_ratio = expected_return / (annual_downside + 1e-8)

log_returns = np.log1p(portfolio_returns_scaled)
log_cum_returns = np.cumsum(log_returns)
rolling_max_log = np.maximum.accumulate(log_cum_returns)
drawdown_log = log_cum_returns - rolling_max_log
drawdown = np.expm1(drawdown_log)
max_drawdown = np.min(drawdown)

positive_pct = np.mean(portfolio_returns_scaled > 0) * 100
avg_pos = np.mean(portfolio_returns_scaled[portfolio_returns_scaled > 0])
avg_neg = np.mean(np.abs(portfolio_returns_scaled[portfolio_returns_scaled < 0]))
avg_p_to_l = avg_pos / (avg_neg + 1e-8)

# --------------------------
# Save .npy
# --------------------------
np.save("portfolio_returns_fixed_40.npy", portfolio_returns_scaled.values)
np.save("cum_returns_fixed_40.npy", np.cumsum(portfolio_returns_scaled.values))
np.save("drawdown_fixed_40.npy", drawdown.values)
np.save("dates_fixed_40.npy", dates.values)
np.save("log_returns_fixed_40.npy", log_returns.values)
np.save("vol_adjustments_fixed_40.npy", vol_adjustment.values)
np.save("metrics_fixed_40.npy", np.array([
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    expected_return,
    volatility,
    positive_pct,
    avg_p_to_l
]))

# --------------------------
# Print Results
# --------------------------
print("Fixed Allocation Strategy (Volatility-Scaled):")
print(f"Expected Return (Annualized): {expected_return:.4f}")
print(f"Volatility (Annualized): {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")
print(f"% Positive Returns: {positive_pct:.2f}%")
print(f"Avg Gain / Avg Loss: {avg_p_to_l:.4f}")
print("Max log cumulative sum:", np.max(np.cumsum(log_returns)))