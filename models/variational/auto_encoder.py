"""Variational Autoencoder (VAE) """

import torch
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """Joint autoencoder that inherits from PyTorch's nn.Module class."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through the ClusterAutoencoder."""
        mu, logvar = self.encoder(x)
        z = self._reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def get_reconstruction_loss(self, x, x_recon):
        """Compute the reconstruction loss."""
        # Binary cross-entropy loss
        #x = torch.sigmoid(x)
        #x_recon = torch.sigmoid(x_recon)
        #return F.binary_cross_entropy(x_recon, x, reduction='sum')

        # Mean squared error loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        #normalize by the number of pixels
        return recon_loss / x.size(0)

    def get_kl_divergence(self, mu, logvar):
        """Compute the Kullback-Leibler divergence."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def feature_array(self, data):
        """Get the feature map from the encoder."""
        mu, _ = self.encoder(data.to('cuda'))
        feature_array = mu.detach().cpu().numpy()
        feature_array = (feature_array - feature_array.mean()
                         ) / feature_array.std()
        return feature_array
