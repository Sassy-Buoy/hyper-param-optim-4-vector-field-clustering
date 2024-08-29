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

    def get_loss(self, x, x_recon):
        """Compute the loss."""

        # Binary cross-entropy loss
        # x = torch.sigmoid(x)
        # x_recon = torch.sigmoid(x_recon)
        # return F.binary_cross_entropy(x_recon, x, reduction='sum')

        # Mean squared error loss
        return F.mse_loss(x_recon, x, reduction='sum')

        # Weighted MSE loss
        # weights = torch.ones(x.size())
        # return torch.mean(weights * (x_recon - x) ** 2)

    def feature_array(self, data):
        """Get the feature map from the encoder."""
        feature_array = self.encoder(data)
        feature_array = feature_array.detach().numpy()
        feature_array = feature_array.reshape(feature_array.shape[0], -1)
        return feature_array
