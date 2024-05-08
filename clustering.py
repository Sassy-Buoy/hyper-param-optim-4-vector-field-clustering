import torch
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


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
        loss = silhouette_score(x, clusters)
        return loss


def silhouette_score_mod(x, clusters):
    """Compute the silhouette score."""
    return silhouette_score(x, clusters)


def std_distance_centroids(x, clusters):
    """ Compute the standard deviation of the distances between the centroids of the clusters."""
    centroids = []

    for cluster in clusters:
        cluster_points = x[clusters == cluster]
        centroid = cluster_points.mean(dim=0)
        centroids.append(centroid)
    
    centroids = torch.stack(centroids)
    distances = torch.cdist(centroids, centroids)
    std_distance = distances.std()

    return std_distance


def silhouette_score_mod(x, clusters):
    """Compute the silhouette score but modified so that ."""


