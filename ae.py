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

    def forward(self, x):
        """Forward pass through the encoder."""
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
        for layer in self.layers:
            x = layer(x)
        return x


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
        """Compute the binary cross-entropy loss."""
        # normalize the data to be in the range [0, 1]
        x = torch.sigmoid(x)
        x_recon = torch.sigmoid(x_recon)
        return F.binary_cross_entropy(x_recon, x, reduction='sum')
