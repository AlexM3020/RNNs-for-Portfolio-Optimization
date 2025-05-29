import numpy as np
import torch
from model import PortfolioLSTM
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------
# Load preprocessed data
# --------------------------
data = torch.load('preprocessed_data.pt', weights_only=False)
X_tensor = data['X_tensor']   # torch.Tensor: input sequences
data = torch.load('future_returns_pt.pt', weights_only=False)
future_returns = data['future_returns']
# Predict weights on all data
# --- Load Trained Model ---
model = PortfolioLSTM(input_size=8)  # Match input_size with training
model.load_state_dict(torch.load("lstm_model.pth"))
model.eval()
with torch.no_grad():
    predicted_weights = model(X_tensor)  # (n_days, n_assets)

min_length = min(len(predicted_weights), len(future_returns))
predicted_weights = predicted_weights[:min_length]
future_returns = future_returns[:min_length]

# Portfolio daily returns
original_returns = data['aligned_returns']  # shape: (n_days, n_assets)
original_returns = original_returns[:len(predicted_weights)]
portfolio_returns = np.sum(predicted_weights.cpu().numpy() * original_returns, axis=1)

#portfolio_returns = torch.sum(predicted_weights * future_returns, dim=1).cpu().numpy()
portfolio_returns = np.clip(portfolio_returns, -0.2, 0.1)
print("Return range:", np.min(portfolio_returns), np.max(portfolio_returns))

# --- 1. Expected Return (Annualized) ---
mean_daily_return = np.mean(portfolio_returns)
expected_return = mean_daily_return * 252

# --- 2. Standard Deviation (Annualized) ---
daily_std = np.std(portfolio_returns)
annual_std = daily_std * np.sqrt(252)

# --- 3. Sharpe Ratio ---
sharpe_ratio = expected_return / (annual_std + 1e-8)

# --- 4. Downside Deviation (Annualized) ---
negative_returns = portfolio_returns[portfolio_returns < 0]
downside_std = np.std(negative_returns)
annual_downside = downside_std * np.sqrt(252)

# --- 5. Sortino Ratio ---
sortino_ratio = expected_return / (annual_downside + 1e-8)


# --- 6. Maximum Drawdown (log-scale, no exp) ---
log_returns = np.log1p(portfolio_returns)
log_cum_returns = np.cumsum(log_returns)
rolling_max_log = np.maximum.accumulate(log_cum_returns)
drawdown_log = log_cum_returns - rolling_max_log
drawdown = np.expm1(drawdown_log)  # convert log drawdown back to % space
max_drawdown = np.min(drawdown)
#if np.isnan(max_drawdown):
    #max_drawdown = 0.0

# --- 7. % Positive Returns ---
positive_pct = np.mean(portfolio_returns > 0) * 100

# --- 8. Avg P / Avg L ---
avg_pos = np.mean(portfolio_returns[portfolio_returns > 0])
avg_neg = np.mean(np.abs(portfolio_returns[portfolio_returns < 0]))
avg_p_to_l = avg_pos / (avg_neg + 1e-8)



# --------------------------
# Generate time axis (working days)
# --------------------------
dates = pd.date_range(start="2010-01-01", end="2024-12-31", freq='B')[:len(portfolio_returns)]
dates = np.array(dates, dtype='datetime64[D]')  # numpy-friendly format

# --------------------------
# Save variables to .npy files
# --------------------------
np.save("portfolio_returns_windows.npy", portfolio_returns)
np.save("cum_returns_windows.npy", np.cumsum(portfolio_returns))
np.save("drawdown_windows.npy", drawdown)
np.save("predicted_weights_windows.npy", predicted_weights.cpu().numpy())
np.save("dates_windows.npy", dates)

# --- Print All ---
# Debug print
#print("Sample log_returns:", log_returns[:5])
#print("Sample log_cum_returns:", log_cum_returns[:5])
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
