"""Variational Autoencoder (VAE) """

import torch
from torch import nn
import torch.nn.functional as F
from conv2dsame import Conv2dSame


class Encoder(nn.Module):
    """Encoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self,
                 num_layers,
                 poolsize,
                 channels,
                 kernel_sizes,
                 dilations,
                 activations):
        super().__init__()
        self.channels = channels
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Conv2dSame(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=1,
                dilation=dilations[i]))
            self.layers.append(activations[i]())
            self.layers.append(nn.MaxPool2d(
                poolsize[i], ceil_mode=True))
        # For VAE: layers to output mean and log variance
        self.fc_mu = nn.Linear(channels[-1], channels[-1])
        self.fc_logvar = nn.Linear(channels[-1], channels[-1])

    def forward(self, x):
        """Forward pass through the encoder."""
        for layer in self.layers:
            x = layer(x)
        # Flatten and pass through the linear layers
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """Decoder class that mirrors the structure of the Encoder using convolution transpose."""

    def __init__(self, encoder):
        super().__init__()
        self.layers = nn.ModuleList()
        # Reverse the encoder layers
        layers_list = list(encoder.layers)
        layers_list.reverse()
        for layer in layers_list:
            if isinstance(layer, nn.Conv2d):
                self.layers.append(Conv2dSame(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation))
            elif isinstance(layer, nn.MaxPool2d):
                self.layers.append(nn.Upsample(scale_factor=layer.kernel_size))
            else:
                self.layers.append(layer)

    def forward(self, x):
        """Forward pass through the decoder."""
        # Reshape the input to be 4D
        x = x.view(x.size(0), -1, 1, 1)
        for layer in self.layers:
            x = layer(x)
        return x


class VaDE(nn.Module):
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
        # return F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
        return F.mse_loss(x_recon, x, reduction='sum')

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

    def get_loss(self, x):
        """Compute the VaDE loss function."""
        x_recon, mu, logvar, z = self.forward(x)
        recon_loss = self.get_reconstruction_loss(x, x_recon)
        kl_divergence = self.get_kl_divergence(mu, logvar, z)
        loss = (recon_loss + kl_divergence)
        return loss
