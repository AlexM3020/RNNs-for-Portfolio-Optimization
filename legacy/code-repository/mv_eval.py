import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from mv_model import MinimumVariancePortfolio

# --------------------------
# Load preprocessed data
# --------------------------
data = torch.load('preprocessed_data.pt', weights_only=False)
returns_df = data['returns']  # pandas DataFrame: daily asset returns

data = torch.load('future_returns_pt.pt', weights_only=False)
future_returns = data['future_returns']
aligned_returns = data['aligned_returns']

# Match shapes
min_length = min(len(future_returns), len(aligned_returns))
future_returns = future_returns[:min_length]
aligned_returns = aligned_returns[:min_length]

# --------------------------
# Fit MV model on all data
# --------------------------
mv_model = MinimumVariancePortfolio(num_assets=aligned_returns.shape[1])
mv_model.fit(aligned_returns.values)

# Predict same weights for each day
predicted_weights = mv_model.forward(torch.zeros((len(aligned_returns), 1)))

# Convert to numpy for further calculations
predicted_weights_np = predicted_weights.numpy()

# Ensure that predicted_weights has the same number of assets as aligned_returns
# Now, predicted_weights_np should be of shape (num_days, num_assets)
# No need for np.repeat anymore as it's already in the correct shape

# --------------------------
# Portfolio daily returns
# --------------------------
# Now, we can calculate the daily portfolio returns using the correct shapes
portfolio_returns = np.sum(predicted_weights_np * aligned_returns.values, axis=1)

# Clip portfolio returns
portfolio_returns = np.clip(portfolio_returns, -0.2, 0.1)

# --------------------------
# Evaluation Metrics
# --------------------------
mean_daily_return = np.mean(portfolio_returns)
expected_return = mean_daily_return * 252

daily_std = np.std(portfolio_returns)
annual_std = daily_std * np.sqrt(252)

sharpe_ratio = expected_return / (annual_std + 1e-8)

negative_returns = portfolio_returns[portfolio_returns < 0]
downside_std = np.std(negative_returns)
annual_downside = downside_std * np.sqrt(252)

sortino_ratio = expected_return / (annual_downside + 1e-8)

log_returns = np.log1p(portfolio_returns)
log_cum_returns = np.cumsum(log_returns)
rolling_max_log = np.maximum.accumulate(log_cum_returns)
drawdown_log = log_cum_returns - rolling_max_log
drawdown = np.expm1(drawdown_log)
max_drawdown = np.min(drawdown)

positive_pct = np.mean(portfolio_returns > 0) * 100
avg_pos = np.mean(portfolio_returns[portfolio_returns > 0])
avg_neg = np.mean(np.abs(portfolio_returns[portfolio_returns < 0]))
avg_p_to_l = avg_pos / (avg_neg + 1e-8)

# --------------------------
# Save
# --------------------------
dates = pd.date_range(start="2010-01-01", periods=len(portfolio_returns), freq='B')
np.save("mv_portfolio_returns.npy", portfolio_returns)
np.save("mv_cum_returns.npy", np.cumsum(portfolio_returns))
np.save("mv_drawdown.npy", drawdown)
np.save("mv_weights.npy", predicted_weights_np)
np.save("mv_dates.npy", dates)

# --------------------------
# Print
# --------------------------
print("Max daily return:", np.max(portfolio_returns))
print("Min daily return:", np.min(portfolio_returns))
print("Max log cumulative sum:", np.max(log_cum_returns))
print(f"Expected Return (Annualized): {expected_return:.4f}")
print(f"Standard Deviation (Annualized): {annual_std:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Downside Deviation (Annualized): {annual_downside:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")
print(f"% Positive Returns: {positive_pct:.2f}%")
print(f"Avg Positive / Avg Negative: {avg_p_to_l:.4f}")
