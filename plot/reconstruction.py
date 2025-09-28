"""Plot a random reconstruction from the test set."""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import pathlib as pl
import io
import torch


def reconstruction(model, data, i=None):
    """Plots a random reconstruction from the test set.
    Args:
        model: torch.nn.Module
            The trained model to use for reconstruction.
        data: torch.utils.data.Dataset
            The dataset to use for reconstruction.
        i: int, optional
            The index of the sample to reconstruct. If None, a random sample is chosen.
    Returns:
        None
    """
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
    ax[0].set_title("Input")
    ax[1].imshow(out_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
    ax[1].set_title("Reconstruction")
    plt.show()


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

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        arr = np.array(img)
        buf.close()

        images.append(arr)
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


