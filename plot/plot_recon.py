"""Plot a random reconstruction from the test set."""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import pathlib as pl
import io
import torch


def plot_recon(model, data, i=None):
    """Plot a random reconstruction from the test set."""
    if i is None:
        i = np.random.randint(0, len(data))
    in_sim = data[i : i + 1]
    out_sim = model(in_sim)
    in_sim = in_sim.detach().numpy()
    in_sim = in_sim.reshape(
        in_sim.shape[0], in_sim.shape[2], in_sim.shape[3], in_sim.shape[1]
    )
    if isinstance(out_sim, tuple):
        out_sim = out_sim[0]
    out_sim = out_sim.detach().numpy()
    out_sim = out_sim.reshape(
        out_sim.shape[0], out_sim.shape[2], out_sim.shape[3], out_sim.shape[1]
    )
    _, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(in_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
    ax[1].imshow(out_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
    plt.show()


def reconstruct(model, data, i=None):
    """Return a random reconstruction from the test set for plotting."""
    if i is None:
        i = np.random.randint(0, len(data))
    in_sim = data[i : i + 1].to("cuda")
    out_sim = model(in_sim)
    in_sim = in_sim.cpu()
    in_sim = in_sim.detach().numpy()
    in_sim = in_sim.reshape(
        in_sim.shape[0], in_sim.shape[2], in_sim.shape[3], in_sim.shape[1]
    )
    if isinstance(out_sim, tuple):
        out_sim = out_sim[0]
    out_sim = out_sim.cpu()
    out_sim = out_sim.detach().numpy()
    out_sim = out_sim.reshape(
        out_sim.shape[0], out_sim.shape[2], out_sim.shape[3], out_sim.shape[1]
    )

    return in_sim, out_sim


def fig_to_array(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    arr = np.array(img)
    buf.close()
    return arr


def reconstruct_gif(data_path, original):
    """make a gif of the reconstructions per epoch"""
    # Load the data
    tensor_files = sorted(pl.Path(data_path).glob("*.pth"))
    recons = [torch.load(f) for f in tensor_files]

    # Create a GIF from the reconstructions
    images = []
    for i in range(len(recons)):
        in_sim = original.unsqueeze(0)
        out_sim = recons[i].cpu()
        in_sim = in_sim.reshape(
            in_sim.shape[0], in_sim.shape[2], in_sim.shape[3], in_sim.shape[1]
        )
        out_sim = out_sim.reshape(
            out_sim.shape[0], out_sim.shape[2], out_sim.shape[3], out_sim.shape[1]
        )
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        ax[0].imshow(in_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
        ax[0].axis("off")

        ax[1].imshow(out_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
        ax[1].axis("off")

        plt.suptitle(f"Epoch {i + 1}")
        plt.tight_layout()

        images.append(fig_to_array(fig))
        plt.close()

    # Save the GIF
    imageio.mimsave("reconstruction.gif", images, fps=10)


if __name__ == "__main__":
    import os
    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    from models.lit_model import LitModel, DataModule

    version = 18
    path = f"lightning_logs/version_{version}/"

    with open(os.path.join(path, "hparams.yaml")) as f:
        hparams = yaml.safe_load(f)

    ckpt = [f for f in os.listdir(path) if f.endswith(".ckpt")][1]

    litmodel = LitModel.load_from_checkpoint(
        checkpoint_path=os.path.join(path, ckpt), config=hparams["config"]
    )

    data_module = DataModule(batch_size=64, num_workers=2)
    data_module.setup()
    test_set = data_module.test_data
    model = litmodel.model
    model.eval()

    reconstruct_gif(
        data_path="lightning_logs/version_18/reconstructions",
        original=data_module.test_data[1],
    )


def plot_latent(model, data):
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


def latent_gif(data_path):
    """
    Create a GIF of the latent space over epochs.
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
        images.append(fig_to_array(fig))
        plt.close()
    imageio.mimsave("latent_space.gif", images, fps=10)


if __name__ == "__main__":
    latent_gif(data_path="lightning_logs/version_18/latent_space_per_epoch")
