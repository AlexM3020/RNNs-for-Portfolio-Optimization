import torch
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize

# --------------------------
# Load preprocessed returns
# --------------------------
data = torch.load('preprocessed_data.pt', weights_only=False)
returns_df = data['returns']  # pandas DataFrame: daily asset returns

# --------------------------
# Define rolling windows
# --------------------------
rolling_windows = [
    ('2010-01-01', '2012-01-01', '2012-01-01', '2014-01-01'),
    ('2010-01-01', '2014-01-01', '2014-01-01', '2016-01-01'),
    ('2010-01-01', '2016-01-01', '2016-01-01', '2018-01-01'),
    ('2010-01-01', '2018-01-01', '2018-01-01', '2020-01-01'),
    ('2010-01-01', '2020-01-01', '2020-01-01', '2022-01-01'),
    ('2010-01-01', '2022-01-01', '2022-01-01', '2024-01-01'),
    ('2010-01-01', '2024-01-01', '2024-01-01', '2024-12-31'),
]

# --------------------------
# Helper: MV optimization
# --------------------------
def min_var_weights(cov_matrix):
    n = cov_matrix.shape[0]
    init_weights = np.ones(n) / n

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    bounds = tuple((0, 1) for _ in range(n))

    def portfolio_variance(w):
        return w.T @ cov_matrix @ w

    result = minimize(portfolio_variance, init_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    return result.x if result.success else init_weights

# --------------------------
# Rolling MV optimization
# --------------------------
all_mv_results = []

for i, (train_start, train_end, test_start, test_end) in enumerate(rolling_windows):
    print(f"\nMV Window {i+1}: Train {train_start} → {train_end}, Test {test_start} → {test_end}")

    train_mask = (returns_df.index >= train_start) & (returns_df.index < train_end)
    test_mask = (returns_df.index >= test_start) & (returns_df.index < test_end)

    train_returns = returns_df.loc[train_mask]
    test_returns = returns_df.loc[test_mask]

    if len(train_returns) < 100 or len(test_returns) < 100:
        print("Skipping window due to insufficient data")
        continue

    cov_matrix = train_returns.cov().values
    mv_weights = min_var_weights(cov_matrix)

    test_portfolio_returns = test_returns @ mv_weights
    test_cum_returns = (1 + test_portfolio_returns).cumprod()

    sharpe_ratio = (
        test_portfolio_returns.mean() / (test_portfolio_returns.std() + 1e-8)
    ) * np.sqrt(252)

    all_mv_results.append((test_start, test_end, sharpe_ratio))
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

# --------------------------
# Final summary
# --------------------------
print("\nMV Rolling Sharpe Summary:")
for start, end, sharpe in all_mv_results:
    print(f"{start} → {end} : Sharpe = {sharpe:.4f}")
