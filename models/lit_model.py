"""Lightning module for training and evaluation."""

import os

import torch
import lightning as L
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import imageio

from models import vanilla, variational
from cluster_acc import purity, adj_rand_index
from plot import plot_umap

sim_arr_tensor = torch.load('./data/sim_arr_tensor.pt')


class LitAE(L.LightningModule):
    """Lightning module for training and evaluation."""

    def __init__(self, hyperparameters: dict, cluster: bool = False, gif: bool = False):
        super().__init__()
        self.lr = hyperparameters["lr"]
        self.cluster = cluster
        encoder = vanilla.Encoder(hyperparameters["num_layers"],
                                  hyperparameters["poolsize"],
                                  hyperparameters["channels"],
                                  hyperparameters["kernel_sizes"],
                                  hyperparameters["dilations"],
                                  hyperparameters["activations"])
        decoder = vanilla.Decoder(encoder)
        self.model = vanilla.AutoEncoder(encoder, decoder)
        self.gif = gif
        self.frames = []  # For storing images to make GIF

    def training_step(self, batch, batch_idx):
        x_recon = self.model(batch)
        loss = self.model.get_loss(batch, x_recon)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_recon = self.model(batch)
        loss = self.model.get_loss(batch, x_recon)
        self.log("val_loss", loss)
        if self.cluster or self.gif:
            feature_array = self.model.feature_array(sim_arr_tensor)
            hdbscan_model = HDBSCAN(min_cluster_size=3,
                                    min_samples=3,
                                    cluster_selection_epsilon=0.96)
            labels = hdbscan_model.fit_predict(feature_array)
        if self.cluster:
            self.log_dict({"purity_score": purity(labels),
                           "ARI": adj_rand_index(labels)})
        if self.gif:
            plot_umap(feature_array, labels)
            fname = f"umap_frame_{self.current_epoch}_{batch_idx}.png"
            plt.savefig(fname)
            plt.close()

        # Store image path for GIF generation later
        self.frames.append(fname)

    def test_step(self, batch, batch_idx):
        x_recon = self.model(batch)
        loss = self.model.get_loss(batch, x_recon)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def on_validation_end(self):
        """If gif is True, create a gif from the saved frames."""
        if self.gif and self.frames:
            images = []
            for filename in self.frames:
                images.append(imageio.imread(filename))
            imageio.mimsave('latent_space_evolution.gif', images, fps=2)

            # Clean up image files after GIF creation
            for file in self.frames:
                os.remove(file)
            self.frames.clear()


class LitVAE(L.LightningModule):
    """Lightning module for training and evaluation."""

    def __init__(self, hyperparameters: dict, cluster: bool = False):
        super().__init__()
        self.lr = hyperparameters["lr"]
        self.cluster = cluster
        encoder = variational.Encoder(hyperparameters["num_layers"],
                                      hyperparameters["poolsize"],
                                      hyperparameters["channels"],
                                      hyperparameters["kernel_sizes"],
                                      hyperparameters["dilations"],
                                      hyperparameters["activations"])
        decoder = variational.Decoder(encoder)
        self.model = variational.AutoEncoder(encoder, decoder)
        self.beta = hyperparameters["beta"] if "beta" in hyperparameters else 1.0

    def training_step(self, batch, batch_idx):
        x_recon, mu, logvar = self.model(batch)
        reconstruction_loss = self.model.get_reconstruction_loss(
            batch, x_recon)
        kl_divergence = self.model.get_kl_divergence(mu, logvar)
        self.log("train_reconstruction_loss", reconstruction_loss)
        self.log("train_kl_divergence", kl_divergence)
        loss = reconstruction_loss + self.beta * kl_divergence
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_recon, mu, logvar = self.model(batch)
        reconstruction_loss = self.model.get_reconstruction_loss(
            batch, x_recon)
        kl_divergence = self.model.get_kl_divergence(mu, logvar)
        self.log("val_reconstruction_loss", reconstruction_loss)
        self.log("val_kl_divergence", kl_divergence)
        loss = reconstruction_loss + kl_divergence
        self.log("val_loss", loss)
        if self.cluster:
            purity_score, adj_rand = self.cluster_acc()
            self.log_dict({"purity_score": purity_score, "ARI": adj_rand})

    def test_step(self, batch, batch_idx):
        x_recon, mu, logvar = self.model(batch)
        reconstruction_loss = self.model.get_reconstruction_loss(
            batch, x_recon)
        kl_divergence = self.model.get_kl_divergence(mu, logvar)
        self.log("test_reconstruction_loss", reconstruction_loss)
        self.log("test_kl_divergence", kl_divergence)
        loss = reconstruction_loss + kl_divergence
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def cluster_acc(self):
        """ Calculate adjusted rand index and purity scores."""
        feature_array = self.model.feature_array(sim_arr_tensor)
        hdbscan_model = HDBSCAN(min_cluster_size=5,
                                cluster_selection_epsilon=0.5)
        labels = hdbscan_model.fit_predict(feature_array)
        return purity(labels), adj_rand_index(labels)


class LitVaDE(L.LightningModule):
    """Lightning module for training and evaluation."""

    def __init__(self, hyperparameters: dict, cluster: bool = False):
        super().__init__()
        self.lr = hyperparameters["lr"]
        self.cluster = cluster
        encoder = variational.Encoder(hyperparameters["num_layers"],
                                      hyperparameters["poolsize"],
                                      hyperparameters["channels"],
                                      hyperparameters["kernel_sizes"],
                                      hyperparameters["dilations"],
                                      hyperparameters["activations"])
        decoder = variational.Decoder(encoder)
        self.model = variational.DeepEmbedding(encoder, decoder,
                                               hyperparameters["n_clusters"])
        self.beta = hyperparameters["beta"] if "beta" in hyperparameters else 1.0

    def training_step(self, batch, batch_idx):
        x_recon, mu, logvar, z = self.model(batch)
        reconstruction_loss = self.model.get_reconstruction_loss(
            batch, x_recon)
        kl_divergence = self.model.get_kl_divergence(mu, logvar, z)
        self.log("train_reconstruction_loss", reconstruction_loss)
        self.log("train_kl_divergence", kl_divergence)
        loss = reconstruction_loss + self.beta * kl_divergence
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_recon, mu, logvar, z = self.model(batch)
        reconstruction_loss = self.model.get_reconstruction_loss(
            batch, x_recon)
        kl_divergence = self.model.get_kl_divergence(mu, logvar, z)
        self.log("val_reconstruction_loss", reconstruction_loss)
        self.log("val_kl_divergence", kl_divergence)
        loss = reconstruction_loss + kl_divergence
        self.log("val_loss", loss)
        if self.cluster:
            purity_score, adj_rand = self.cluster_acc()
            self.log_dict({"purity_score": purity_score, "ARI": adj_rand})

    def test_step(self, batch, batch_idx):
        x_recon, mu, logvar, z = self.model(batch)
        reconstruction_loss = self.model.get_reconstruction_loss(
            batch, x_recon)
        kl_divergence = self.model.get_kl_divergence(mu, logvar, z)
        self.log("test_reconstruction_loss", reconstruction_loss)
        self.log("test_kl_divergence", kl_divergence)
        loss = reconstruction_loss + kl_divergence
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def cluster_acc(self):
        """ Calculate adjusted rand index and purity scores."""
        labels = self.model.classify(sim_arr_tensor.to('cuda'))
        labels = labels.detach().cpu().numpy()
        return purity(labels), adj_rand_index(labels)
