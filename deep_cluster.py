import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KmeansLayer(torch.nn.Module):
    """Kmeans layer that inherits from PyTorch's nn.Module class."""

    def __init__(self, n_clusters):
        super(KmeansLayer, self).__init__()
        self.n_clusters = n_clusters

    def forward(self, feature_array):
        """Forward pass through the K-means layer."""
        # fit K-means model
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(feature_array)

        return kmeans.labels_

    def _get_loss_ss(self, feature_array, labels):
        """Calculate the silhouette score."""
        ss = silhouette_score(feature_array, labels)
        return 1/ss

    def _get_loss_std(self, feature_array, labels):
        """Calculate the standard deviation of the cluster centers."""
        centroids = []
        # fill the array with the centroids
        for i in labels:
            centroids.append(np.mean(feature_array[labels == i], axis=0))
        # initialize the array that will contain the distances
        distances = []
        # fill the array with the distances
        for i in range(len(centroids)):
            for j in range(len(centroids)):
                distances.append(np.linalg.norm(centroids[i] - centroids[j]))
        # return the std of the distances
        return np.std(distances)


class Classifier(torch.nn.Module):
    """Classifier that inherits from PyTorch's nn.Module class."""

    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass through the classifier."""
        x = self.fc(x)
        x = self.softmax(x)
        # based on the softmax output, assign to 1 of the clusters
        labels = torch.argmax(x, dim=1)
        return x


class JointAutoencoder(torch.nn.Module):
    """Joint autoencoder that inherits from PyTorch's nn.Module class."""
    def __init__(self, encoder, decoder, cluster, full_set, train_set):
        super(JointAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cluster = cluster
        self.full_set = full_set
        self.epochs = 100
        self.batch_size = 64
        self.data = train_set
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        """Forward pass through the ClusterAutoencoder."""
        x, indices_list = self.encoder(x)
        with torch.no_grad():
            feature_array, _ = self.encoder(self.full_set)
            feature_array = feature_array.view(feature_array.size(0), -1)
            feature_array = feature_array.cpu().detach().numpy()
        labels = self.cluster(feature_array)
        x = self.decoder(x, indices_list)

        return x, labels

    def _get_reconstruction_loss(self, x, x_hat):
        """Calculate the reconstruction loss."""
        return torch.nn.functional.mse_loss(x_hat, x, reduction='mean')

    def _get_weighted_mse_loss(self, x, cluster_centers):
        pass

    def _get_loss(self, x):
        """Calculate the loss function."""
        x_hat, labels = self(x)
        reconstruction_loss = self._get_reconstruction_loss(x, x_hat)
        with torch.no_grad():
            feature_array, _ = self.encoder(self.full_set)
            feature_array = feature_array.view(feature_array.size(0), -1)
            feature_array = feature_array.cpu().detach().numpy()
        cluster_loss = self.cluster._get_loss_std(feature_array, labels)
        print(reconstruction_loss, 0.5*cluster_loss)
        return reconstruction_loss + 0.5*cluster_loss

    def feature_array(self):
        with torch.no_grad():
            feature_array, _ = self.encoder(self.full_set)
            feature_array = feature_array.view(feature_array.size(0), -1)
            feature_array = feature_array.cpu().detach().numpy()
            labels = self.cluster(feature_array)
        return feature_array, labels

    def train_model(self, dataloader=None, device=torch.device('cuda:0')):
        """Train the autoencoder."""
        if dataloader is None:
            dataloader = torch.utils.data.DataLoader(
                self.data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=12)
        self.to(device)
        self.train()

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            running_loss = 0.0

            for batch in dataloader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                loss = self._get_loss(batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * batch.size(0)
            epoch_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} Loss: {epoch_loss:.4f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == 5:
                    print(f"Early stopping after {epoch+1} epochs.")
                    break

    def evaluate_model(self, dataloader, device=torch.device('cuda:0')):
        """Evaluate the autoencoder."""
        self.to(device)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                loss = self._get_loss(batch)
                total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def cross_val(self, n_splits=5):
        """Perform cross-validation on the autoencoder."""
        torch.backends.cudnn.benchmark = True
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
