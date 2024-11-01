"""Variational Deep Embedding (VaDE) """

import torch
from torch import nn
import torch.nn.functional as F


class DeepEmbedding(nn.Module):
    """Variational Deep Embedding that inherits from PyTorch's nn.Module class."""

    def __init__(self, encoder, decoder, n_clusters):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_clusters = n_clusters
        self.latent_dim = encoder.channels[-1]

        # GMM parameters
        self.pi_prior = nn.Parameter(torch.ones(
            n_clusters) / n_clusters, requires_grad=True)
        self.mu_prior = nn.Parameter(torch.randn(
            n_clusters, self.latent_dim), requires_grad=True)
        self.logvar_prior = nn.Parameter(torch.randn(
            n_clusters, self.latent_dim), requires_grad=True)

    def _reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through the VaDE."""
        mu, logvar = self.encoder(x)  # Encode
        z = self._reparameterize(mu, logvar)  # Reparameterize
        x_recon = self.decoder(z)  # Decode
        return x_recon, mu, logvar, z

    def classify(self, x):
        """Classify the input x into one of the n_clusters."""
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z = self._reparameterize(mu, logvar)
            # p(z|c) = N(z; mu_c, var_c) = exp(-0.5 * (z - mu_c)^2 / var_c) / sqrt(2 * pi * var_c)
            z_expand = z.unsqueeze(1)
            mu_expand = self.mu_prior.unsqueeze(0)
            logvar_expand = self.logvar_prior.unsqueeze(0)
            log_p_z_given_c = -0.5 * \
                (logvar_expand + torch.pow(z_expand -
                 mu_expand, 2) / torch.exp(logvar_expand))
            log_p_z_given_c = torch.sum(log_p_z_given_c, dim=2)
            log_p_z_given_c += torch.log(self.pi_prior + 1e-10)
            return torch.argmax(log_p_z_given_c, dim=1)

    def get_reconstruction_loss(self, x, x_recon):
        """Compute the BCE loss between the input x and the reconstructed x."""
        x = F.sigmoid(x)
        x_recon = F.sigmoid(x_recon)
        return nn.BCELoss(reduction='sum')(x_recon, x)

    def get_kl_divergence(self, mu, logvar, z):
        """Compute the Kullback-Leibler divergence.
        D_kl(q(z|x) || p(z|c)) = -sum(q(z|x) * log(p(z|c) / q(z|x)))"""
        # Prior distribution parameters
        pi_prior = F.softmax(self.pi_prior, dim=0)
        mu_prior = self.mu_prior
        logvar_prior = self.logvar_prior

        # Log of p(z|c) using the LSE trick
        z_expand = z.unsqueeze(1)
        mu_expand = mu_prior.unsqueeze(0)
        logvar_expand = logvar_prior.unsqueeze(0)
        log_p_z_given_c = -0.5 * \
            (logvar_expand + (z_expand - mu_expand) ** 2 / torch.exp(logvar_expand))
        log_p_z_given_c = torch.sum(log_p_z_given_c, dim=2)
        log_p_z_given_c += torch.log(pi_prior + 1e-10)

        # Use log-sum-exp trick for numerical stability
        log_p_z = torch.logsumexp(log_p_z_given_c, dim=1)

        # Log of q(z|x)
        log_q_z_given_x = -0.5 * (logvar + (z - mu) ** 2 / torch.exp(logvar))
        log_q_z_given_x = torch.sum(log_q_z_given_x, dim=1)

        # KL divergence
        kl_divergence = log_q_z_given_x - log_p_z
        loss = torch.sum(kl_divergence)
        if torch.isnan(loss):
            print("KL_div is nan. Using 0 instead.")
            return torch.tensor(0.0, device=mu.device)
        return loss

    def feature_array(self, data):
        """Get the feature map from the encoder."""
        mu, _ = self.encoder(data)
        feature_array = mu.detach().numpy()
        feature_array = (feature_array - feature_array.mean()
                         ) / feature_array.std()
        return feature_array
