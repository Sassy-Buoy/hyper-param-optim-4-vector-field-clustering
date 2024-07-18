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
        # indices_list = []
        for layer in self.layers:
            x = layer(x)

        # Flatten and pass through the linear layers
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def encode(self, x):
        """Encode the input data."""
        for layer in self.layers:
            x = layer(x)
        return x


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
        x = x.view(x.size(0), -1, 1, 1)
        for layer in self.layers:
            x = layer(x)

        return x

    def decode(self, x):
        """Decode the input data."""
        for layer in self.layers:
            x = layer(x)
        return x


class VarAutoEncoder(nn.Module):
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
        return torch.mean((x - x_recon) ** 2)

    def get_kl_divergence(self, mu, logvar):
        """Compute the Kullback-Leibler divergence."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def get_loss(self, x):
        """Compute the VAE loss."""
        x_recon, mu, logvar = self.forward(x)
        return self.get_reconstruction_loss(x, x_recon) + self.get_kl_divergence(mu, logvar)
    
    def reconstruct(self, x):
        """Reconstruct the input data."""
        y = self.encoder.encode(x)
        return self.decoder.decode(y)



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

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through the VaDE."""
        # Encode
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        # Decode
        x_recon = self.decoder(z)

        # GMM responsibilities (q(y|x))
        z_expand = z.unsqueeze(1)  # (batch_size, 1, latent_dim)
        mu_expand = self.mu_prior.unsqueeze(0)  # (1, n_clusters, latent_dim)
        logvar_expand = self.logvar_prior.unsqueeze(
            0)  # (1, n_clusters, latent_dim)
        pi_expand = self.pi_prior.unsqueeze(0)  # (1, n_clusters)

        log_p_z_given_c = -0.5 * \
            (logvar_expand + torch.pow(z_expand -
             mu_expand, 2) / torch.exp(logvar_expand))
        # (batch_size, n_clusters)
        log_p_z_given_c = torch.sum(log_p_z_given_c, dim=2)
        log_p_z_given_c += torch.log(pi_expand)  # (batch_size, n_clusters)

        q_y_given_x = F.softmax(log_p_z_given_c, dim=1)

        return x_recon, z, q_y_given_x

    def get_reconstruction_loss(self, x, x_recon):
        """Compute the reconstruction loss."""
        return torch.mean((x - x_recon) ** 2)

    def get_kl_divergence(self, z, q_y_given_x):
        """Compute the Kullback-Leibler divergence."""
        # KL divergence between q(z|x) and p(z|c)
        z_expand = z.unsqueeze(1)  # (batch_size, 1, latent_dim)
        mu_expand = self.mu_prior.unsqueeze(0)  # (1, n_clusters, latent_dim)
        logvar_expand = self.logvar_prior.unsqueeze(
            0)  # (1, n_clusters, latent_dim)

        log_p_z_given_c = -0.5 * \
            (logvar_expand + torch.pow(z_expand -
             mu_expand, 2) / torch.exp(logvar_expand))
        # (batch_size, n_clusters)
        log_p_z_given_c = torch.sum(log_p_z_given_c, dim=2)
        log_p_z_given_c += torch.log(self.pi_prior + 1e-10)  # (n_clusters)

        log_q_y_given_x = torch.log(q_y_given_x)
        kl_div = torch.sum(
            q_y_given_x * (log_q_y_given_x - log_p_z_given_c), dim=1)

        # KL divergence between q(y|x) and p(y)
        log_q_y_given_x = torch.log(q_y_given_x + 1e-10)
        kl_div_y = torch.sum(q_y_given_x * log_q_y_given_x, dim=1)

        return torch.sum(kl_div) + torch.sum(kl_div_y)

    def get_loss(self, x):
        """Compute the VaDE loss function."""
        x_recon, z, q_y_given_x = self.forward(x)
        return self.get_reconstruction_loss(x, x_recon) + self.get_kl_divergence(z, q_y_given_x)

    def predict(self, x):
        """Predict the cluster assignment."""
        _, _, q_y_given_x = self.forward(x)
        return torch.argmax(q_y_given_x, dim=1)
