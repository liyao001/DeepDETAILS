import torch
from torch import nn


class RMSLELoss(nn.Module):
    """
    Root Mean Square Log Error

    """

    def __init__(self, root: bool = True):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.root = root

    def forward(self, pred: torch.Tensor, actual: torch.Tensor):
        # shape of self.mse(): batch, clusters, seq_len
        # shape of self.mse().sum(axis=-1): batch, clusters
        mse = (
            self.mse(
                torch.log(torch.clamp_min(pred, -0.999) + 1), torch.log(actual + 1)
            )
            .sum(axis=-1)
            .mean()
        )
        return torch.sqrt(mse) if self.root else mse


def off_diagonal(x: torch.Tensor):
    """Return a flattened view of the off-diagonal elements of a square matrix

    Parameters
    ----------
    x : torch.Tensor
        A square matrix (correlation matrix)
        Shape: n, m, where n==m

    Returns
    -------
    torch.Tensor
        n*n-n
    """
    n, m = x.shape
    if n != m:
        raise ValueError(f"Expected a square matrix, got shape {tuple(x.shape)}")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def corrcoef_stable(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise Pearson correlation matrix, safer for (near) zero-variance rows.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    eps : float
        Min value to avoid nans
    """
    x = x - x.mean(dim=-1, keepdim=True)
    std = (x.square().mean(dim=-1) + eps).sqrt()
    x = x / std.unsqueeze(-1)
    corr = (x @ x.transpose(-1, -2)) / x.shape[-1]
    return corr.clamp(-1.0, 1.0)


def mean_sq_offdiag_corr(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean squared off-diagonal Pearson correlation of the rows of ``x``."""
    return off_diagonal(corrcoef_stable(x, eps=eps)).pow(2).mean()
