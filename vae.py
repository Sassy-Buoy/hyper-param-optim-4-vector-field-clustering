"""Variational Autoencoder (VAE) implementation in PyTorch."""

import torch
from torch import nn
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
                poolsize[i], ceil_mode=True))#, return_indices=True))

        # For VAE: layers to output mean and log variance
        self.fc_mu = nn.Linear(channels[-1], channels[-1])
        self.fc_logvar = nn.Linear(channels[-1], channels[-1])

    def forward(self, x):
        """Forward pass through the encoder."""
        #indices_list = []
        for layer in self.layers:
            x = layer(x)
            """if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)"""

        # Flatten and pass through the linear layers
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar#, indices_list


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
                """self.layers.append(nn.MaxUnpool2d(layer.kernel_size,
                                                  layer.stride,
                                                  layer.padding))"""
            else:
                self.layers.append(layer)

    def forward(self, x):#, indices_list):
        """Forward pass through the decoder."""
        x = x.view(x.size(0), -1, 1, 1)
        for layer in self.layers:
            x = layer(x)
            """if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)"""

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
        #mu, logvar, indices_list = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)#, indices_list)
        return x_reconstructed, mu, logvar

    def get_reconstruction_loss(self, x, x_reconstructed):
        """Compute the reconstruction loss."""
        return torch.mean((x - x_reconstructed) ** 2)

    def get_kl_divergence(self, mu, logvar):
        """Compute the Kullback-Leibler divergence."""
        #kl_loss = nn.KLDivLoss(reduction='sum')
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def get_loss(self, x):
        """Compute the VAE loss."""
        x_reconstructed, mu, logvar = self.forward(x)
        reconstruction_loss = self.get_reconstruction_loss(x, x_reconstructed)
        kl_divergence = self.get_kl_divergence(mu, logvar)
        return reconstruction_loss + kl_divergence
