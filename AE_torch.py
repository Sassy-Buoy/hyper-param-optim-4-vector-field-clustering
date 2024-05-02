import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.cluster import DBSCAN


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
            self.layers.append(nn.MaxPool2d(poolsize[i], ceil_mode=True))

    def forward(self, x):
        """Forward pass through the encoder."""
        for layer in self.layers:
            print(x.size(), layer)
            x = layer(x)
        # check if the output has the correct shape
        if x.size() != (1, 3, 1, 1):
            raise ValueError("Output shape does not match expected shape." + str(x.size()))
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
                self.layers.append(nn.ConvTranspose2d(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size,
                    padding=layer.padding,
                    dilation=layer.dilation))
            elif isinstance(layer, nn.MaxPool2d):
                self.layers.append(nn.Upsample(scale_factor=layer.kernel_size))
            else:
                self.layers.append(layer)

    def forward(self, x):
        """Forward pass through the decoder."""
        for layer in self.layers:
            print(x.size(), layer)
            x = layer(x)
        # check if the output has the correct shape
        if x.size() != (1, 3, 80, 80):
            raise ValueError("Output shape does not mutch expected shape." + str(x.size()))
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
                 paddings,
                 activations,
                 epochs,
                 batch_size,
                 learning_rate,
                 data):
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
        x = self.encoder(x)
        x = self.decoder(x)
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
            #epoch_loss = running_loss / len(dataloader)
            #print(f"Epoch {epoch+1}/{self.epochs} Loss: {epoch_loss:.4f}")

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
        #print(f"Validation Loss: {avg_loss:.4f}")
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
                self.data, sampler=train_sampler)
            val_loader = torch.utils.data.DataLoader(
                self.data, sampler=val_sampler)

            self.train_model(train_loader)
            val_loss = self.evaluate_model(val_loader)
            val_losses.append(val_loss)
        return val_losses
