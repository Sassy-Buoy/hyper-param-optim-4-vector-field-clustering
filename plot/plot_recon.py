"""Plot a random reconstruction from the test set."""
import numpy as np
import matplotlib.pyplot as plt


def random_recon(model, data):
    """Plot a random reconstruction from the test set."""
    i = np.random.randint(0, len(data))
    in_sim = data[i: i + 1]
    out_sim = model(in_sim)
    in_sim = in_sim.detach().numpy()
    in_sim = in_sim.reshape(
        in_sim.shape[0], in_sim.shape[2], in_sim.shape[3], in_sim.shape[1])
    if isinstance(out_sim, tuple):
        out_sim = out_sim[0]
    out_sim = out_sim.detach().numpy()
    out_sim = out_sim.reshape(
        out_sim.shape[0], out_sim.shape[2], out_sim.shape[3], out_sim.shape[1])
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(in_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
    ax[1].imshow(out_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
    plt.show()

    labels = np.load('data/labels.npy')
    print(labels[i])
