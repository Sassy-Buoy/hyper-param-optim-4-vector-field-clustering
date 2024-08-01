import numpy as np
import matplotlib.pyplot as plt


def random_recon(model, test_data):
    i = np.random.randint(0, len(test_data))
    in_sim = test_data[i: i + 1]
    in_sim = in_sim.to('cuda:0')
    out_sim = model(in_sim)
    in_sim = in_sim.detach().to('cpu').numpy()
    in_sim = in_sim.reshape(
        in_sim.shape[0], in_sim.shape[2], in_sim.shape[3], in_sim.shape[1])
    out_sim = out_sim.detach().to('cpu').numpy()
    out_sim = out_sim.reshape(
        out_sim.shape[0], out_sim.shape[2], out_sim.shape[3], out_sim.shape[1])
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(in_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
    ax[1].imshow(out_sim[0, ..., 2], vmin=-1, vmax=1, cmap="RdBu")
    plt.show()
