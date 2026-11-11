import torch


def sharpe_ratio_loss(weights, returns, epsilon=1e-6):
    portfolio_returns = torch.sum(weights * returns, dim=1)  # (batch,)

    # Remove NaNs/Infs safely
    portfolio_returns = torch.where(torch.isfinite(portfolio_returns), portfolio_returns,
                                    torch.zeros_like(portfolio_returns))

    mean = torch.mean(portfolio_returns)
    std = torch.std(portfolio_returns)

    sharpe = mean / (std + epsilon)

    # Optionally clamp to avoid exploding losses
    if not torch.isfinite(sharpe):
        return torch.tensor(1e6, device=weights.device, requires_grad=True)

    return -sharpe

