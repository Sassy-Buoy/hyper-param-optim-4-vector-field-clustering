import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self,
                 num_layers,
                 poolsize,
                 channels,
                 kernel_sizes,
                 dilations,
                 paddings,
                 activations):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                padding=paddings[i],
                dilation=dilations[i]))
            self.layers.append(activations[i]())
            self.layers.append(nn.MaxPool2d(poolsize[i]))

    def forward(self, x):
        """Forward pass through the encoder."""
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Decoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self,
                 num_layers,
                 poolsize,
                 channels,
                 kernel_sizes,
                 dilations,
                 paddings,
                 activations):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            self.layers.append(nn.ConvTranspose2d(in_channels=channels[i],
                                                  out_channels=channels[i + 1],
                                                  kernel_size=kernel_sizes[i],
                                                  stride=poolsize[i],
                                                  padding=paddings[i],
                                                  dilation=dilations[i]))
            self.layers.append(activations[i]())

    def forward(self, x):
        """Forward pass through the decoder."""
        for layer in self.layers:
            x = layer(x)
        return x


class Autoencoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self,
                 num_layers,
                 poolsize,
                 channels,
                 kernel_sizes,
                 dilations,
                 paddings,
                 activations):
        super().__init__()
        # Encoder
        self.encoder = Encoder(num_layers,
                               poolsize,
                               channels,
                               kernel_sizes,
                               dilations,
                               paddings,
                               activations)
        # Decoder
        self.decoder = Decoder(num_layers,
                               poolsize,
                               channels,
                               kernel_sizes,
                               dilations,
                               paddings,
                               activations)

    def forward(self, x):
        """Forward pass through the autoencoder."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def _get_reconstruction_loss(self, batch):
        """Compute the reconstruction loss."""
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss


def Searchspace(trial, input_dim, output_dim):
        """ define the hyperparameter search space."""
        num_layers = trial.suggest_int('num_layers', 2, 5)
        if num_layers == 2:
            poolsize = trial.suggest_categorical('poolsize_2', [[5, 16], [16, 5],
                                                                   [8, 10], [10, 8],
                                                                   [4, 20], [20, 4],
                                                                   [2, 40], [40, 2]])
        elif num_layers == 3:
            poolsize = trial.suggest_categorical(
                'poolsize_3', [[2, 2, 20], [2, 20, 2], [20, 2, 2],
                             [4, 4, 5], [4, 5, 4], [5, 4, 4],
                             [2, 5, 8], [2, 8, 5], [5, 2, 8],
                             [5, 8, 2], [8, 2, 5], [8, 5, 2],
                             [2, 4, 10], [2, 10, 4], [4, 2, 10],
                             [4, 10, 2], [10, 2, 4], [10, 4, 2]])
        elif num_layers == 4:
            poolsize = trial.suggest_categorical(
                'poolsize_4', [[2, 2, 2, 10], [2, 2, 10, 2], [2, 10, 2, 2], [10, 2, 2, 2],
                            [2, 2, 4, 5], [2, 2, 5, 4], [2, 4, 2, 5],
                            [2, 4, 5, 2],[2, 5, 2, 4], [2, 5, 4, 2],
                            [4, 2, 2, 5], [4, 2, 5, 2], [4, 5, 2, 2],
                            [5, 2, 2, 4], [5, 2, 4, 2], [5, 4, 2, 2]])
        elif num_layers == 5:
            poolsize = trial.suggest_categorical(
                'poolsize_5', [[2, 2, 2, 2, 5], [2, 2, 2, 5, 2], [2, 2, 5, 2, 2],
                             [2, 5, 2, 2, 2], [5, 2, 2, 2, 2]])

        channels = [input_dim]
        channels.append([trial.suggest_int(
            f'filters_{i}', 1, 12) for i in range(num_layers - 1)])
        channels.append(output_dim)

        kernel_sizes = [trial.suggest_int(
            f'kernel_size_{i}', 2, 24) for i in range(num_layers)]

        dilations = [trial.suggest_categorical(
            f'dilation_{i}', [0, 2, 4]) for i in range(num_layers)]

        paddings = [x * (y - 1)//2 for x, y in zip(dilations, kernel_sizes)]

        activations = [trial.suggest_categorical(
            f'activation_{i}', [nn.Softplus, nn.SELU, nn.SiLU, nn.Tanh]) for i in range(num_layers)]

        return [num_layers, poolsize, channels, kernel_sizes, dilations, paddings, activations]
