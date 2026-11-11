"""The original single-layer LSTM allocator, without import-time execution."""

import torch
from torch import nn


class PortfolioLSTM(nn.Module):
    """Map a batch of historical windows to fully invested long-only weights."""

    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_assets: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return torch.softmax(self.linear(hidden[-1]), dim=1)


def sharpe_ratio_loss(weights: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Negative batch Sharpe on raw next-day returns (zero risk-free rate)."""
    if weights.shape != returns.shape or weights.ndim != 2 or len(weights) < 2:
        raise ValueError("Weights and returns must match and contain at least two observations.")
    if not torch.isfinite(weights).all() or not torch.isfinite(returns).all():
        raise ValueError("Non-finite weights or returns.")
    portfolio = (weights * returns).sum(dim=1)
    return -portfolio.mean() / (portfolio.std() + 1e-6)
