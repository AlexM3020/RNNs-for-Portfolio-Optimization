import numpy as np
import torch
from model import PortfolioLSTM
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Load preprocessed data
# --------------------------
window_number = 4  # scegli la finestra da valutare

data = torch.load(f'data_windows/data1', weights_only=False)
X_tensor = data['X_tensor']
future_returns = data['future_returns']
original_returns = data['aligned_returns']


# Predict weights on all data
model = PortfolioLSTM(input_size=8)  # Match input_size with training
model = torch.load("models/final_lstm_full_model.pt", weights_only=False)
model.eval()

with torch.no_grad():
    predicted_weights = model(X_tensor)

# Trim to minimum length
min_length = min(len(predicted_weights), len(future_returns), len(original_returns))
predicted_weights = predicted_weights[:min_length]
future_returns = future_returns[:min_length]
original_returns = original_returns.iloc[:min_length]

# --------------------------
# Volatility Scaling
# --------------------------
sigma_target = 0.10  # 10% annual target volatility
window = 50

# Convert returns to DataFrame
returns_df = pd.DataFrame(original_returns, index=original_returns.index)
rolling_std = returns_df.rolling(window=window).std()
annualized_vol = rolling_std * np.sqrt(252)
vol_adjustment = sigma_target / (annualized_vol + 1e-6)

# Apply volatility scaling to weights
scaled_weights = predicted_weights.cpu().numpy() * vol_adjustment.values
scaled_weights = np.nan_to_num(scaled_weights)

# Re-normalize to ensure weights sum to 1

row_sums = scaled_weights.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0  # avoid division by zero
scaled_weights /= row_sums


# Portfolio daily returns
portfolio_returns = np.sum(scaled_weights * original_returns.values, axis=1)
#portfolio_returns = np.clip(portfolio_returns, -0.2, 0.1)  # Optional clipping

print("Return range:", np.min(portfolio_returns), np.max(portfolio_returns))

# --------------------------
# Metrics
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
# Generate time axis
# --------------------------
dates = pd.date_range(start="2010-01-01", end="2024-12-31", freq='B')[:len(portfolio_returns)]
dates = np.array(dates, dtype='datetime64[D]')

# Recalculate as multiplicative growth
cumulative_returns_dl = 1 + np.cumsum(portfolio_returns)

# --------------------------
# Save .npy
# --------------------------
np.save("portfolio_returns_windows.npy", portfolio_returns)
np.save("cum_returns_windows.npy", cumulative_returns_dl)
np.save("drawdown_windows.npy", drawdown)
np.save("predicted_weights_windows.npy", scaled_weights)
np.save("dates_windows.npy", dates)
# New saves for additional plots
np.save("log_returns_windows.npy", log_cum_returns)  # For log-scale analysis
np.save("annualized_vol_windows.npy", annualized_vol.values)  # For volatility plots
np.save("vol_adjustments_windows.npy", vol_adjustment.values)  # For weight analysis

# Save metrics as numpy array (for easy loading)
np.save("metrics_windows.npy", np.array([
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    expected_return,
    annual_std,
    positive_pct,
    avg_p_to_l
]))
# --------------------------
# Print results
# --------------------------
print("Max daily return:", np.max(portfolio_returns))
print("Min daily return:", np.min(portfolio_returns))
print("Max log cumulative sum:", np.max(np.cumsum(log_returns)))
print(f"Expected Return (Annualized): {expected_return:.4f}")
print(f"Standard Deviation (Annualized): {annual_std:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Downside Deviation (Annualized): {annual_downside:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")
print(f"% Positive Returns: {positive_pct:.2f}%")
print(f"Avg Positive / Avg Negative: {avg_p_to_l:.4f}")
# --------------------------
# Visualize Log Cumulative Returns and Drawdowns
# --------------------------
plt.figure(figsize=(14, 6))
dates = pd.to_datetime(np.load("dates_windows.npy")) + pd.DateOffset(months=6)
# Plot log cumulative returns
plt.subplot(2, 1, 1)
plt.plot(dates, log_cum_returns, label='Log Cumulative Return', color='blue')
plt.plot(dates, rolling_max_log, label='Rolling Max', color='orange', linestyle='--')
plt.fill_between(dates, log_cum_returns, rolling_max_log, where=log_cum_returns < rolling_max_log,
                 color='red', alpha=0.3, label='Drawdown')
plt.title("Cumulative Returns and Drawdowns")
plt.legend()
plt.grid(True)

# Plot drawdown (in % scale)
plt.subplot(2, 1, 2)
plt.plot(dates, drawdown * 100, color='red')
plt.title("Drawdown (%) from Peak")
plt.grid(True)

plt.tight_layout()
plt.show()
