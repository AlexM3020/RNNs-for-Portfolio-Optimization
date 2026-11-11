"""Long-only classical baselines fitted exclusively to past observations."""

import numpy as np
from scipy.optimize import minimize


def optimized_weights(history: np.ndarray, strategy: str) -> np.ndarray:
    """Minimum variance or genuine maximum diversification ratio."""
    covariance = np.cov(history, rowvar=False) + np.eye(history.shape[1]) * 1e-10
    asset_vol = np.sqrt(np.diag(covariance))
    initial = np.full(history.shape[1], 1 / history.shape[1])

    def objective(weights):
        variance = float(weights @ covariance @ weights)
        if strategy == "minimum_variance":
            return variance * 1e4
        if strategy == "maximum_diversification":
            return -float(weights @ asset_vol) / np.sqrt(max(variance, 1e-16))
        raise ValueError(f"Unknown optimized strategy: {strategy}")

    result = minimize(objective, initial, method="SLSQP", bounds=[(0, 1)] * len(initial),
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                      options={"ftol": 1e-10, "maxiter": 200})
    if not result.success:
        raise RuntimeError(f"{strategy} optimization failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()
