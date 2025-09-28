"""Plotting functions for latent space visualization."""

import torch
import matplotlib.pyplot as plt
import imageio
import pathlib as pl
import io
import numpy as np
from PIL import Image


def latent_space(model, data):
    """Plot the latent space of the model."""
    with torch.no_grad():
        z = model.encode(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=data.y, cmap="viridis")
    plt.colorbar()
    plt.title("Latent Space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.show()


def latent_gif(data_path, save_path="latent_space.gif"):
    """
    Create a GIF of the latent space over epochs.
    Args:
        data_path: str
            Path to the directory containing latent space tensors saved per epoch.
        save_path: str
            Path to save the output GIF.
    Returns:
        None
    """
    tensor_files = sorted(pl.Path(data_path).glob("*.pth"))
    images = []
    for f in tensor_files:
        z = torch.load(f)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(z[:, 0], z[:, 1], cmap="viridis")
        plt.colorbar(scatter)
        plt.title(f"Latent Space - Epoch {f.stem}")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        arr = np.array(img)
        buf.close()

        images.append(arr)
        plt.close()
    imageio.mimsave(save_path, images, fps=10)


if __name__ == "__main__":
    latent_gif(data_path="lightning_logs/version_18/latent_space_per_epoch", save_path="latent_space.gif")
