import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
# --------------------------
# MV Model Definition
# --------------------------
class MinimumVariancePortfolio:
    def __init__(self, num_assets):
        self.num_assets = num_assets
        self.weights = np.ones(num_assets) / num_assets

    def fit(self, returns):
        cov = np.cov(returns.T)

        def variance(w):
            return w.T @ cov @ w

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * self.num_assets

        result = minimize(variance, self.weights, bounds=bounds, constraints=constraints)
        self.weights = result.x if result.success else self.weights

    def get_weights(self, num_days):
        return np.tile(self.weights, (num_days, 1))

# --------------------------
# Load and Prepare Data
# --------------------------
data_future = torch.load("future_returns_pt.pt", weights_only=False)
future_returns = data_future['future_returns'].numpy()
aligned_returns = data_future['aligned_returns'].values

min_len = min(len(future_returns), len(aligned_returns))
aligned_returns = aligned_returns[:min_len]

dates = pd.date_range(start="2010-01-01", periods=min_len, freq='B')

# --------------------------
# Fit MV Strategy
# --------------------------
mv = MinimumVariancePortfolio(num_assets=aligned_returns.shape[1])
mv.fit(aligned_returns)
weights = mv.get_weights(len(aligned_returns))

# --------------------------
# Compute Raw Portfolio Returns
# --------------------------
raw_returns = np.sum(weights * aligned_returns, axis=1)
raw_returns = np.clip(raw_returns, -0.2, 0.1)

# --------------------------
# Volatility Scaling
# --------------------------
target_vol = 0.10
window = 50
rolling_std = pd.Series(raw_returns).rolling(window).std() * np.sqrt(252)
vol_adjustment = target_vol / (rolling_std + 1e-8)
scaled_returns = raw_returns * vol_adjustment
scaled_returns[:window] = 0

# --------------------------
# Metrics
# --------------------------
log_returns = np.log1p(scaled_returns)
log_cum = np.cumsum(log_returns)
rolling_max = np.maximum.accumulate(log_cum)
drawdown = np.expm1(log_cum - rolling_max)

expected_return = np.mean(scaled_returns) * 252
volatility = np.std(scaled_returns) * np.sqrt(252)
sharpe = expected_return / (volatility + 1e-8)

neg = scaled_returns[scaled_returns < 0]
sortino = expected_return / (np.std(neg) * np.sqrt(252) + 1e-8)
max_dd = np.min(drawdown)
positive_pct = np.mean(scaled_returns > 0) * 100
avg_gain = np.mean(scaled_returns[scaled_returns > 0])
avg_loss = np.mean(np.abs(scaled_returns[scaled_returns < 0]))
p_to_l = avg_gain / (avg_loss + 1e-8)

# --------------------------
# Save Outputs
# --------------------------
np.save("mv_portfolio_returns.npy", scaled_returns)
np.save("mv_cum_returns.npy", np.cumsum(scaled_returns))
np.save("mv_drawdown.npy", drawdown)
np.save("mv_weights.npy", weights)
np.save("mv_dates.npy", dates)
np.save("mv_metrics.npy", np.array([
    sharpe, sortino, max_dd, expected_return,
    volatility, positive_pct, p_to_l
]))

# --------------------------
# Print Summary
# --------------------------
print(f"Expected Return (Annualized): {expected_return:.4f}")
print(f"Volatility (Annualized): {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print("Max log cumulative sum:", np.max(np.cumsum(log_returns)))
print(f"Sortino Ratio: {sortino:.4f}")
print(f"Maximum Drawdown: {max_dd:.4f}")
print(f"% Positive Returns: {positive_pct:.2f}%")
print(f"Avg Gain / Avg Loss: {p_to_l:.4f}")
