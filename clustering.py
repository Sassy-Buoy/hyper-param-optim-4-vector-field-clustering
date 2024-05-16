import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


def silhouette_score_mod(feature_array, labels):
    """Compute the silhouette score."""
    pass


def std_distance_centroids(feature_array, labels):
    """ Compute the standard deviation of the distances between the centroids of the clusters."""

    centroids = []
    # Fill the array with the centroids
    for i in labels:
        centroids.append(np.mean(feature_array[labels == i], axis=0))

    distances = []
    # Fill the array with the distances
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            distances.append(np.linalg.norm(centroids[i] - centroids[j]))

    return np.std(distances)


class KmeansLayer(torch.nn.Module):
    """Kmeans layer that inherits from PyTorch's nn.Module class."""

    def __init__(self, n_clusters):
        super().__init__()
        self.n_clusters = n_clusters

    def forward(self, x, feature_array):
        """Forward pass through the K-means layer."""
        # fit K-means model
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(feature_array)
        return kmeans.labels_

    def get_loss(self, x, feature_array, labels):
        """Compute the clustering loss."""
        # loss = 1/silhouette_score(x, labels)
        loss = std_distance_centroids(feature_array, labels)
        return loss


class DBSCANLayer(torch.nn.Module):
    """DBSCAN layer that inherits from PyTorch's nn.Module class."""

    def __init__(self, eps, min_samples):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def forward(self, x, feature_array):
        """Forward pass through the DBSCAN layer."""
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan.fit_predict(self.train_set)
        labels = dbscan.fit_predict(feature_array)
        return labels

    def get_loss(self, x, feature_array, labels):
        """Compute the clustering loss."""
        # Compute the clustering loss
        # loss = 1/(silhouette_score(feature_array, labels))
        loss = std_distance_centroids(feature_array, labels)
        return loss


class Classifier(torch.nn.Module):
    """Classifier that inherits from PyTorch's nn.Module class."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass through the classifier."""
        x = self.fc(x)
        x = self.softmax(x)
        # based on the softmax output, treating it like a binary number, assign the classes.
        for i in range(x.size(0)):
            x[i] = torch.tensor([1 if j == x[i].max() else 0 for j in x[i]])
        return labels

    def _get_loss(self, x, labels):
        """Calculate the cross-entropy loss."""
        return torch.nn.functional.cross_entropy(x, labels)

    def _get_weighted_mse_loss(self, x):
        labels = self.cluster(x)
        # weight matrix
        weight_matrix = torch.zeros((x.size(0), x.size(0)))
        for i in range(x.size(0)):
            for j in range(x.size(0)):
                if labels[i] == labels[j]:
                    weight_matrix[i][j] = np.exp(-np.linalg.norm(
                        x[i] - x[j])**2)/len(x == labels[i])
                else:
                    weight_matrix[i][j] = 0
        # calculate the weighted mse loss
        return torch.nn.functional.mse_loss(x, x, weight=weight_matrix, reduction='mean')
