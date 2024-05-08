# defines the Autoencoder class that inherits from PyTorch's nn.Module class

import math
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.cluster import DBSCAN


class Conv2dSame(nn.Conv2d):
    """
    SOURCE: https://github.com/pytorch/captum/blob/optim-wip/captum/optim/models/_common.py#L144
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions.
    TODO: Replace with torch.nn.Conv2d when support for padding='same'
    is in stable version
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """
        See nn.Conv2d for more details on the possible arguments:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:

           in_channels (int): The expected number of channels in the input tensor.
           out_channels (int): The desired number of channels in the output tensor.
           kernel_size (int or tuple of int): The desired kernel size to use.
           stride (int or tuple of int, optional): The desired stride for the
               cross-correlation.
               Default: 1
           padding (int or tuple of int, optional): This value is always set to 0.
               Default: 0
           dilation (int or tuple of int, optional): The desired spacing between the
               kernel points.
               Default: 1
           groups (int, optional): Number of blocked connections from input channels
               to output channels. Both in_channels and out_channels must be divisable
               by groups.
               Default: 1
           bias (bool, optional): Whether or not to apply a learnable bias to the
               output.
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """
        Calculate the required padding for a dimension.

        Args:

            i (int): The specific size of the tensor dimension requiring padding.
            k (int): The size of the Conv2d weight dimension.
            s (int): The Conv2d stride value for the dimension.
            d (int): The Conv2d dilation value for the dimension.

        Returns:
            padding_vale (int): The calculated padding value.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.tensor): The input tensor to apply 2D convolution to.

        Returns
            x (torch.Tensor): The input tensor after the 2D convolution was applied.
        """
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=kh, s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=kw, s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


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
        if x.size() != (1, 3, 1, 1):
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
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)
        # check if the output has the correct shape
        if x.size() != (1, 3, 80, 80):
            raise ValueError(
                "Output shape does not mutch expected shape." + str(x.size()))
        return x


class DBSCANLayer(torch.nn.Module):
    """DBSCAN layer that inherits from PyTorch's nn.Module class."""

    def __init__(self, eps, min_samples):
        super(DBSCANLayer, self).__init__()
        self.eps = eps
        self.min_samples = min_samples

    def forward(self, x):
        """Forward pass through the DBSCAN layer."""
        # Convert PyTorch tensor to numpy array
        x_np = x.detach().cpu().numpy()

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = dbscan.fit_predict(x_np)

        # Convert clusters to PyTorch tensor
        clusters_tensor = torch.from_numpy(clusters).to(x.device)

        return clusters_tensor

    def _clustering_loss(self, x):
        """Compute the clustering loss."""
        clusters = self.forward(x)
        # Compute the clustering loss
        loss = torch.tensor(0.0)
        return loss


class Autoencoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self,
                 num_layers,
                 poolsize,
                 channels,
                 kernel_sizes,
                 dilations,
                 activations,
                 epochs,
                 batch_size,
                 learning_rate,
                 data):
        super().__init__()
        # Encoder
        self.encoder = Encoder(num_layers=num_layers,
                               poolsize=poolsize,
                               channels=channels,
                               kernel_sizes=kernel_sizes,
                               dilations=dilations,
                               activations=activations)
        # Decoder
        self.decoder = Decoder(self.encoder)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # data and dataloader
        self.data = data

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the autoencoder."""
        x, indices_list = self.encoder(x)
        x = self.decoder(x, indices_list)
        return x

    def _get_reconstruction_loss(self, batch):
        """Compute the reconstruction loss."""
        x = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        return loss

    def train_model(self, dataloader, device=torch.device('cuda:0')):
        """Train the autoencoder."""
        self.to(device)
        self.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch in dataloader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                loss = self._get_reconstruction_loss(batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * batch.size(0)
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {epoch_loss:.4f}")

    def evaluate_model(self, dataloader, device=torch.device('cuda:0')):
        """Evaluate the autoencoder."""
        self.to(device)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                loss = self._get_reconstruction_loss(batch)
                total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def cross_val(self, n_splits=5):
        """Perform cross-validation on the autoencoder."""
        kf = KFold(n_splits=n_splits, shuffle=True)
        val_losses = []
        for fold, (train_index, val_index) in enumerate(kf.split(self.data)):
            print(f"Fold {fold+1}/{n_splits}")
            train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_index)

            train_loader = torch.utils.data.DataLoader(
                self.data,
                sampler=train_sampler,
                batch_size=self.batch_size,
                num_workers=12)
            val_loader = torch.utils.data.DataLoader(
                self.data,
                sampler=val_sampler,
                batch_size=self.batch_size,
                num_workers=12)

            self.train_model(train_loader)
            val_loss = self.evaluate_model(val_loader)
            val_losses.append(val_loss)
        return val_losses
