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

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through the ClusterAutoencoder."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def get_reconstruction_loss(self, x, x_recon):
        """Compute the reconstruction loss."""
        # Binary cross-entropy loss
        #x = torch.sigmoid(x)
        #x_recon = torch.sigmoid(x_recon)
        #return F.binary_cross_entropy(x_recon, x, reduction='sum')

        # Mean squared error loss
        return F.mse_loss(x_recon, x, reduction='sum')

    def get_kl_divergence(self, mu, logvar):
        """Compute the Kullback-Leibler divergence."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def get_loss(self, x):
        """Compute the VAE loss."""
        x_recon, mu, logvar = self.forward(x)
        recon_loss = self.get_reconstruction_loss(x, x_recon)
        kl_loss = self.get_kl_divergence(mu, logvar)
        return recon_loss + 1000*kl_loss, recon_loss, 1000*kl_loss

    def feature_array(self, data):
        """Get the feature map from the encoder."""
        mu, _ = self.encoder(data.to('cuda'))
        feature_array = mu.detach().to('cpu').numpy()
        feature_array = (feature_array - feature_array.mean()
                         ) / feature_array.std()
        return feature_array
