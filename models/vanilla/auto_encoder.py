"""Vanilla Autoencoder class."""

from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """Autoencoder that inherits from PyTorch's nn.Module class."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Forward pass through the ClusterAutoencoder."""
        y = self.encoder(x)
        x_recon = self.decoder(y)
        return x_recon

    def get_loss(self, x):
        """Compute the binary cross-entropy loss."""
        x_recon = self.forward(x)

        # Binary cross-entropy loss
        # x = torch.sigmoid(x)
        # x_recon = torch.sigmoid(x_recon)
        # return F.binary_cross_entropy(x_recon, x, reduction='sum')

        # Mean squared error loss
        return F.mse_loss(x_recon, x, reduction='sum')

        # Weighted MSE loss
        # weights = torch.ones(x.size())
        # return torch.mean(weights * (x_recon - x) ** 2)
