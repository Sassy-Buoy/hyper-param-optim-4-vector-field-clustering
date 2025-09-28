"""Autoencoder and Variational Autoencoder models for 2D image data."""

from math import ceil

import torch
from torch import Tensor, nn


class AutoEncoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self, layers: list[dict]):
        """
        Initializes the Encoder and Decoder networks.
        The Decoder is built in reverse order of the Encoder.
        Args: layers (list[dict]): List of dictionaries defining the layers.
            Each dictionary has the key type(str). The possible types are:
            -"Conv" : Convolutional layer
            -"Linear" : Fully connected layer
            -"Flatten" : Flatten layer takes argument "in_size" and "in_channels".
        """
        super(AutoEncoder, self).__init__()

        self.encoder = []
        self.decoder = []

        for i, layer in enumerate(layers):
            # Insert the encoder layers
            if layer["type"] == "Conv":
                self.encoder += self._conv_block(
                    in_channels=layer["in_channels"],
                    out_channels=layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                )
                # Insert the decoder layers in reverse order
                self.decoder = (
                    self._conv_block(
                        in_channels=layer["out_channels"],
                        out_channels=layer["in_channels"],
                        kernel_size=layer["kernel_size"],
                        stride=layer["stride"],
                        transpose=True,
                    )
                    + self.decoder
                )
            elif layer["type"] == "Flatten":
                self.encoder.append(nn.Flatten())
                self.decoder.insert(
                    0,
                    nn.Unflatten(
                        1,
                        (
                            layer["in_channels"],
                            layer["in_size"],
                            layer["in_size"],
                        ),
                    ),
                )
            elif layer["type"] == "Linear":
                self.encoder += self._linear_block(
                    in_features=layer["in_features"],
                    out_features=layer["out_features"],
                    activation=layer["activation"],
                )
                self.decoder = (
                    self._linear_block(
                        in_features=layer["out_features"],
                        out_features=layer["in_features"],
                        activation=layer["activation"],
                    )[::-1]
                    + self.decoder
                )

        # Replace the last activation function in the decoder with Tanh
        self.decoder.pop(-1)
        self.decoder.append(nn.Tanh())

        # convert the lists to nn.Sequential
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def _conv_block(
        self, in_channels, out_channels, kernel_size, stride, transpose=False
    ):
        """
        Creates a convolutional block.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            transpose (bool): If True, creates a transposed convolutional layer.
        Returns:
            list: A list of layers constituting the convolutional block.
        """
        return [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=ceil((kernel_size - stride) / 2),
            )
            if not transpose
            else nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=ceil((kernel_size - stride) / 2),
                output_padding=0
                if in_channels == 256
                else (0 if ((kernel_size - stride) % 2) == 0 else 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]

    def _linear_block(self, in_features, out_features, activation):
        """
        Creates a linear block.
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            activation (str): Type of activation function ("ReLU" or "LeakyReLU").
        Returns:
            list: A list of layers constituting the linear block.
        """
        layers = [nn.Linear(in_features, out_features)]
        if activation == "ReLU":
            layers.append(nn.ReLU())
        elif activation == "LeakyReLU":
            layers.append(nn.LeakyReLU())
        return layers

    def forward(self, x: Tensor):
        """Forward pass through the encoder and decoder. Takes input tensor and returns reconstructed tensor."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """
        Calculates the MSE loss function.
        Args:
            x (Tensor): Original input tensor.
            x_recon (Tensor): Reconstructed tensor.
        Returns:
            Tensor: Computed loss value.
        """
        criterion = nn.MSELoss()
        return criterion(x_recon, x)


class VarAutoEncoder(AutoEncoder):
    """Autoencoder class that inherits from AutoEncoder class."""

    def __init__(self, layers: list[dict]):
        """
        See AutoEncoder for details.
        Beyond that, it additionally initializes :
        Args:
            fc_mean (nn.Linear): Fully connected layer to compute the mean of the latent space.
            fc_log_var (nn.Linear): Fully connected layer to compute the log variance of the latent space.
            latent_dim (int): Dimension of the latent space.
        """
        super(VarAutoEncoder, self).__init__(layers)
        self.latent_dim = layers[-1]["out_features"]
        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.latent_dim, self.latent_dim)
        # Initialize the weights and biases of fc_mean and fc_log_var
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_log_var.weight)
        nn.init.constant_(self.fc_mean.bias, 0)
        nn.init.constant_(self.fc_log_var.bias, -5)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick.
        Args:
            mu (Tensor): Mean of the latent space.
            log_var (Tensor): Log variance of the latent space.
        Returns:
            Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the input.
        Args:
            x (Tensor): Input tensor.
        Returns:
            tuple: Mean and log variance of the latent space.
        """
        x = self.encoder(x)
        mean, log_var = self.fc_mean(x), self.fc_log_var(x)
        return mean, log_var

    def forward(self, x: Tensor):
        """Forward pass through the encoder and decoder.
        Args:
            x (Tensor): Input tensor.
        Returns:
            tuple: Reconstructed tensor, mean, and log variance of the latent space.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

    def get_recon_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Calculates the MSE loss function.
        Args:
            x (Tensor): Original input tensor.
            x_recon (Tensor): Reconstructed tensor.
        Returns:
            Tensor: Computed loss value.
        """
        criterion = nn.MSELoss()
        return criterion(x_recon, x)

    def get_kl_divergence(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Calculates the KL divergence.
        Args:
            mu (Tensor): Mean of the latent space.
            log_var (Tensor): Log variance of the latent space.
        Returns:
            Tensor: Computed KL divergence value.
        """
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_divergence / mu.size(0)

    def latent_space(self, x: Tensor) -> Tensor:
        """Takes input tensor and returns the latent space representation."""
        mean, _ = self.encode(x)
        latent_space = mean.detach()
        return latent_space
