import pandas as pd
import numpy as np
from scipy.optimize import minimize

# --------------------------
# Load daily returns
# --------------------------
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)
n_assets = returns.shape[1]

# Parameters
window = 100
sigma_target = 0.10  # Target annual volatility
lambda_diversity = 0.1  # Tradeoff parameter (higher means more emphasis on diversity/risk)

# Storage for dynamic weights and portfolio returns
weights_dynamic = []
portfolio_returns_md = []

# Constraints for weights: sum to 1, weights >= 0 (long-only)
constraints = ({
    'type': 'eq',
    'fun': lambda w: np.sum(w) - 1
})

# Add bounds, limiting VIX weight to max 0.2
bounds = []
for col in returns.columns:
    if col == "VIX":
        bounds.append((0, 0.2))
    else:
        bounds.append((0, 1))

# Objective function: maximize mean return minus lambda * portfolio variance
# We minimize negative of that for scipy.optimize
def objective(w, mean_vec, cov_mat, lam):
    port_return = w @ mean_vec
    port_var = w @ cov_mat @ w
    return -(port_return - lam * port_var)

# Print average expected returns over rolling windows for debugging
mean_returns_rolling = returns.rolling(window=window).mean().dropna()
avg_means = mean_returns_rolling.mean()
print("Average expected returns over rolling windows:")
print(avg_means)

# Rolling window optimization
for t in range(window, len(returns)):
    ret_window = returns.iloc[t - window:t]
    mean_vec = ret_window.mean().values  # mean returns vector
    cov_mat = ret_window.cov().values  # covariance matrix

    # Initial guess: equal weights
    w0 = np.ones(n_assets) / n_assets

    res = minimize(objective, w0, args=(mean_vec, cov_mat, lambda_diversity),
                   method='SLSQP', bounds=bounds, constraints=constraints)

    if res.success:
        w_opt = res.x
    else:
        # fallback to equal weights if optimization fails
        w_opt = w0

    weights_dynamic.append(w_opt)

    # Compute portfolio return for day t (next day return)
    daily_ret = returns.iloc[t].values @ w_opt
    portfolio_returns_md.append(daily_ret)

# Convert lists to arrays or series
weights_dynamic = np.array(weights_dynamic)
portfolio_returns_md = pd.Series(portfolio_returns_md, index=returns.index[window:])
avg_weights = np.mean(weights_dynamic, axis=0)
print("\nAverage portfolio weights over time:")
for i, w in enumerate(avg_weights):
    print(f"Asset {returns.columns[i]}: {w:.3f}")

# --------------------------
# Volatility Scaling (on dynamic portfolio returns)
# --------------------------
rolling_vol = portfolio_returns_md.rolling(window=window).std() * np.sqrt(252)
vol_adjustment = sigma_target / (rolling_vol + 1e-8)
portfolio_returns_md_scaled = portfolio_returns_md * vol_adjustment
portfolio_returns_md_scaled.fillna(0, inplace=True)

# --------------------------
# Cumulative Returns
# --------------------------
cumulative_returns_md = (1 + portfolio_returns_md).cumprod()
cumulative_returns_md_scaled = (1 + portfolio_returns_md_scaled).cumprod()

# --------------------------
# Metrics — on scaled returns
# --------------------------
mean_daily = portfolio_returns_md_scaled.mean()
std_daily = portfolio_returns_md_scaled.std()
expected_return = mean_daily * 252
volatility = std_daily * np.sqrt(252)

negative_returns = portfolio_returns_md_scaled[portfolio_returns_md_scaled < 0]
downside_std = np.std(negative_returns) * np.sqrt(252)

sharpe_ratio = expected_return / (volatility + 1e-8)
sortino_ratio = expected_return / (downside_std + 1e-8)

log_returns = np.log1p(portfolio_returns_md_scaled)
log_cum_returns = np.cumsum(log_returns)
rolling_max_log = np.maximum.accumulate(log_cum_returns)
drawdown_log = log_cum_returns - rolling_max_log
drawdown = np.expm1(drawdown_log)
max_drawdown = np.min(drawdown)
positive_pct = np.mean(portfolio_returns_md_scaled > 0) * 100
avg_gain = np.mean(portfolio_returns_md_scaled[portfolio_returns_md_scaled > 0])
avg_loss = np.mean(np.abs(portfolio_returns_md_scaled[portfolio_returns_md_scaled < 0]))
avg_p_to_l = avg_gain / (avg_loss + 1e-8)

# --------------------------
# Save or print results
# --------------------------
portfolio_returns_md.to_csv("portfolio_returns_md_dynamic.csv")
cumulative_returns_md.to_csv("md_strategy_cumulative_returns_dynamic.csv")
cumulative_returns_md_scaled.to_csv("md_strategy_cumulative_returns_scaled_dynamic.csv")
np.save("md_drawdown.npy", drawdown)
print("\nVolatility-Scaled Dynamic Mean-Diversity Strategy:")
print(f"Expected Return (Annualized): {expected_return:.4f}")
print(f"Volatility (Annualized): {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")
print(f"% Positive Days: {positive_pct:.2f}%")
print(f"Avg Gain / Avg Loss: {avg_p_to_l:.4f}")
print("Max log cumulative sum:", np.max(np.cumsum(log_returns)))