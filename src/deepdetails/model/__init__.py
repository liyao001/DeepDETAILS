import torch
from torch import nn


class ResidualConv(nn.Module):

    def __init__(self, filters, kernel_size, dilation_rate) -> None:
        super().__init__()
        self.conv = nn.LazyConv1d(
            filters, kernel_size=kernel_size, padding="valid", dilation=dilation_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            input. Shape: batch, channels, seq_len

        Returns
        -------
        out : torch.Tensor
            Shape: batch, channels, seq_len. Note seq_len here may not be equal to the seq_len of x
        """
        conv_out = self.relu(self.conv(x))
        crippling_len = (x.size(2) - conv_out.size(2)) // 2
        out = conv_out + x[:, :, crippling_len:-crippling_len]
        return out


class ResidualConvWithXProjection(nn.Module):

    def __init__(self, filters, kernel_size, dilation_rate) -> None:
        super().__init__()
        self.conv = nn.LazyConv1d(
            filters, kernel_size=kernel_size, padding="valid", dilation=dilation_rate)
        self.relu = nn.ReLU()
        self.inc_dim = nn.LazyConv1d(filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            input. Shape: batch, channels, seq_len

        Returns
        -------
        out : torch.Tensor
            Shape: batch, channels, seq_len. Note seq_len here may not be equal to the seq_len of x
        """
        conv_out = self.relu(self.conv(x))
        crippling_len = (x.size(2) - conv_out.size(2)) // 2
        out = conv_out + self.inc_dim(x[:, :, crippling_len:-crippling_len])
        return out
