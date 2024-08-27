"""Training loop for the autoencoder model with clustering."""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from plot import umap_plot
from cluster_acc import purity, adj_rand_index


def train_cluster(model, train_set, val_set, lr, batch_size, epochs,
                  patience=5, device='cuda'):
    """Train the autoencoder with early stopping based on validation loss stability."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    model.to(device)
    model.train()

    best_val_loss = float('inf')
    epochs_since_improvement = 0
    train_losses = []
    val_losses = []
    purity_scores = []
    adj_rand_scores = []
    image_paths = []

    sim_arr_tensor = torch.load('./data/sim_arr_tensor.pt')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = model.get_loss(batch)
                val_loss += loss.item()
        epoch_val_loss = val_loss / len(val_loader)

        print(f"""Epoch {epoch+1}/{epochs}
              Train Loss: {epoch_train_loss:.4f}
              Val Loss: {epoch_val_loss:.4f}""")
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = model.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"""Early stopping triggered.
                      No improvement in validation loss for {patience} epochs.""")
                break

        # clustering
        feature_array = model.feature_array(sim_arr_tensor)
        kmeans_model = KMeans(n_clusters=15, random_state=42)
        kmeans_model.fit(feature_array)
        labels = kmeans_model.labels_
        purity_scores.append(purity(labels))
        adj_rand_scores.append(adj_rand_index(labels))

        """# make a gif of the clusters
        fig = umap_plot(feature_array, labels)
        plt.savefig(f'./cluster_images/cluster_{epoch}.png')
        plt.close(fig)
        image_paths.append(f'./cluster_images/cluster_{epoch}.png')"""

    model.load_state_dict(best_model)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(purity_scores, label='Purity')
    plt.plot(adj_rand_scores, label='Adjusted Rand Index')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

    """# save the cluster images as a gif
    images = [Image.open(image_path) for image_path in image_paths]
    images[0].save('./cluster_images/cluster.gif', save_all=True,
                   append_images=images[1:], loop=0, duration=100)
    # delete the individual images
    for image_path in image_paths:
        os.remove(image_path)"""
