import numpy as np
import torch
from scipy.optimize import minimize

class MinimumVariancePortfolio:
    def __init__(self, num_assets):
        self.num_assets = num_assets
        self.weights = np.ones(num_assets) / num_assets  # Default to equal weight

    def fit(self, returns: np.ndarray):
        """
        Fit MV model to training return data.
        :param returns: np.ndarray of shape (samples, assets)
        """
        cov_matrix = np.cov(returns.T)

        def portfolio_variance(w):
            return w.T @ cov_matrix @ w

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))

        result = minimize(portfolio_variance, self.weights,
                          method='SLSQP', bounds=bounds,
                          constraints=constraints)

        if result.success:
            self.weights = result.x
        else:
            print("Optimization failed. Using equal weights.")

    def forward(self, x: torch.Tensor):
        """
        Return portfolio weights for input data.
        MV is static (does not depend on x), so this returns the same weights.
        :param x: dummy tensor (not used)
        """
        batch_size = x.shape[0]
        weights_tensor = torch.tensor(self.weights, dtype=torch.float32).unsqueeze(0)
        weights_tensor = weights_tensor.repeat(batch_size, 1)  # repeat weights for each day
        return weights_tensor
