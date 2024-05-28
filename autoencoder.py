""" Autoencoder module that contains the Encoder, Decoder, and ClusterAutoencoder classes. """

import numpy as np
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
                poolsize[i], ceil_mode=True, return_indices=True))

    def forward(self, x):
        """Forward pass through the encoder."""
        indices_list = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)

        # check if the output has the correct shape
        if (x.size()[1], x.size()[2], x.size()[3]) != (12, 1, 1):
            raise ValueError(
                "Output shape does not match expected shape." + str(x.size()))
        return x, indices_list


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
                self.layers.append(nn.MaxUnpool2d(layer.kernel_size,
                                                  layer.stride,
                                                  layer.padding))
            else:
                self.layers.append(layer)

    def forward(self, x, indices_list):
        """Forward pass through the decoder."""
        for layer in self.layers:
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)
        # check if the output has the correct shape
        if (x.size()[1], x.size()[2], x.size()[3]) != (3, 80, 80):
            raise ValueError(
                "Output shape does not mutch expected shape." + str(x.size()))
        return x


class ClusterAutoencoder(torch.nn.Module):
    """Joint autoencoder that inherits from PyTorch's nn.Module class."""

    def __init__(self, encoder, decoder, cluster, train_set):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cluster = cluster
        self.train_set = train_set

    def forward(self, x):
        """Forward pass through the ClusterAutoencoder."""
        x, indices_list = self.encoder(x)
        feature_array = self.encode_data(self.train_set)
        labels = self.cluster(x, feature_array)
        x = self.decoder(x, indices_list)
        return x, labels

    def _get_reconstruction_loss(self, x, x_hat):
        """Calculate the reconstruction loss."""
        return torch.nn.functional.mse_loss(x_hat, x, reduction='mean')

    def weighted_mse_loss(self, x, x_hat, labels):
        """Calculate the weighted mean squared error loss."""
        # weight matrix
        weight_matrix = torch.zeros((x.size(0), x.size(0)))
        x = x.cpu().detach()
        x_hat = x_hat.cpu().detach()
        a, b = 0, 0
        for i in range(x.size(0)):
            for j in range(x.size(0)):
                if labels[i] == labels[j]:
                    weight_matrix[i][j] = np.exp(-np.linalg.norm(
                        x[i] - x[j])**2)
                    a += 1
                else:
                    weight_matrix[i][j] = weight_matrix[i][j] = (1 - np.exp(-np.linalg.norm(
                        x[i] - x[j])**2))/len(x != labels[i])
                    b += 1
        for i in range(x.size(0)):
            for j in range(x.size(0)):
                if labels[i] == labels[j]:
                    weight_matrix[i][j] = weight_matrix[i][j] / a
                else:
                    weight_matrix[i][j] = weight_matrix[i][j] / b

        # calculate the weighted mse loss
        loss = 0
        for i in range(x.size(0)):
            for j in range(x.size(0)):
                loss += weight_matrix[i][j] * \
                    np.linalg.norm(x[i] - x_hat[j])**2
        return (loss / x.size(0)).to('cuda')

    def get_loss(self, x):
        """Calculate the loss function."""
        x_hat, labels = self(x)
        reconstruction_loss = self._get_reconstruction_loss(x, x_hat)
        feature_array = self.encode_data(self.train_set)
        cluster_loss = self.cluster.get_loss(x, feature_array, labels)
        # print(reconstruction_loss, cluster_loss)
        # weighted_mse_loss = self.weighted_mse_loss(x, x_hat, labels)
        # print(weighted_mse_loss)
        # return weighted_mse_loss
        return reconstruction_loss + cluster_loss

    def encode_data(self, data):
        """Return the feature array and labels."""
        with torch.no_grad():
            data = data.to(torch.device('cuda'))
            feature_array, _ = self.encoder(data)
            feature_array = feature_array.view(feature_array.size(0), -1)
            feature_array = feature_array.cpu().detach().numpy()
        return feature_array
