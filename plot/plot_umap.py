""" Plot in 3D the UMAP of the feature array. """

import matplotlib.pyplot as plt
import umap.umap_ as umap


def plot_umap(feature_array, labels):
    """Plot in 3D the UMAP of the feature array."""
    reducer = umap.UMAP(n_components=2)
    reduced_feature_array = reducer.fit_transform(feature_array)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_feature_array[:, 0], reduced_feature_array[:, 1],
                         reduced_feature_array[:, 2], c=labels, cmap='viridis')
    plt.title("UMAP of the feature array")
    plt.colorbar(scatter)
