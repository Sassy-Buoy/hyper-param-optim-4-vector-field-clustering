"""Lightning module for training, evaluation and data handling."""

import os
import lightning as L
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.auto_encoder import AutoEncoder, VarAutoEncoder


class DataModule(L.LightningDataModule):
    """Lightning data module for loading and preprocessing data."""

    def __init__(self, batch_size: int, num_workers: int):
        """Initializes the data module.
        Args:
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of workers for data loaders.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Operations to be performed on every GPU in distributed mode.
        Loads and splits the data into training, validation, and test sets.
        """
        self.data = torch.load("data/sim_arr_tensor.pt")
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )
        self.train_data, self.val_data = train_test_split(
            train_data, test_size=0.2, random_state=42
        )
        self.test_data = test_data

    def train_dataloader(self):
        """Creates the training dataloader."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Creates the validation dataloader."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Creates the test dataloader."""
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LitModel(L.LightningModule):
    """Lightning module for training and evaluation."""

    def __init__(
        self,
        model_type: str,
        config: list,
        learning_rate: float,
        threshold: float | None = None,
    ):
        """Initializes the Lightning module.
        Also initializes the model with specified architecture and saves hyperparameters to hparams.yaml.
        Args:
            model_type (str): Type of model to use ("vanilla" or "variational").
            config (list): Configuration for the model architecture.
            learning_rate (float): Learning rate for the optimizer.
            threshold (float, optional): Threshold for KL divergence in VAE. Defaults to None.
        """
        super().__init__()
        self.model_type = model_type
        if model_type == "vanilla":
            self.model = AutoEncoder(config)
        elif model_type == "variational":
            self.model = VarAutoEncoder(config)
            self.threshold = 8
        self.learning_rate = learning_rate
        # write the model type and configuration to the hparams.yaml file
        self.save_hyperparameters("model_type", "config", "learning_rate")
        if model_type == "variational":
            self.save_hyperparameters("threshold")

    def configure_optimizers(self):
        """Configures the optimizer."""
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def beta_scheduler(self, current_epoch, init_phase=20):
        """
        Scheduler for beta in the VAE loss function.
        Sets beta to 0 for the first 20 epochs, then gradually increase to 1.
        Args:
            current_epoch (int): Current epoch number.
            init_phase (int, optional): Number of epochs with beta=0. Defaults to 20.
        Returns:
            float: Value of beta for the current epoch.
        """
        # beta =0 for the first cycle
        if current_epoch < init_phase:
            beta = 0
        else:
            beta = min(1.0, (current_epoch - init_phase) / 100)
        return beta

    def training_step(self, batch, batch_idx):
        batch = batch.float().to("cuda")
        # torch.autograd.set_detect_anomaly(True)

        if self.model_type == "variational":
            batch_recon, mean, log_var = self.model(batch)
            recon_loss = self.model.get_recon_loss(batch, batch_recon)
            self.log("train_recon_loss", recon_loss, sync_dist=True)

            kl_loss = self.model.get_kl_divergence(mean, log_var)
            self.log("train_kl_loss", kl_loss, sync_dist=True)

            # beta = self.beta_scheduler(self.current_epoch)
            loss = recon_loss + max(kl_loss, self.threshold)

        else:
            loss = self.model.get_loss(batch, self.model(batch))

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.float().to("cuda")
        if self.model_type == "variational":
            batch_recon, mean, log_var = self.model(batch)
            recon_loss = self.model.get_recon_loss(batch, batch_recon)
            self.log("val_recon_loss", recon_loss, sync_dist=True)

            kl_loss = self.model.get_kl_divergence(mean, log_var)
            self.log("val_kl_loss", kl_loss, sync_dist=True)

            # beta = self.beta_scheduler(self.current_epoch)
            loss = recon_loss + max(kl_loss, self.threshold)

        else:
            loss = self.model.get_loss(batch, self.model(batch))

        self.log("val_loss", loss, sync_dist=True)

    def encode_data(self, model, dataloader):
        """Encode the data using the trained VAE or AE.
        Args:
            model (torch.nn.Module): Trained model (VAE or AE).
            dataloader (DataLoader): DataLoader for the data to be encoded.
        Returns:
            torch.Tensor: Encoded data.
        """
        encoded_data = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = batch.float().to("cuda")
                if self.model_type == "variational":
                    encoded_batch = model.latent_space(batch)
                    encoded_data.append(encoded_batch.cpu())
                else:
                    encoded_batch = model.encoder(batch)
                    encoded_data.append(encoded_batch.cpu())
        encoded_data = torch.cat(encoded_data, dim=0)
        return encoded_data

    def on_validation_epoch_end(self):
        """Logs the latent space and reconstructions at the end of each validation epoch."""
        os.makedirs(f"{self.logger.log_dir}/latent_space_per_epoch", exist_ok=True)
        latent_space = self.encode_data(
            self.model,
            DataLoader(self.trainer.datamodule.data, batch_size=64, shuffle=False),
        )
        torch.save(
            latent_space,
            f"{self.logger.log_dir}/latent_space_per_epoch/{self.current_epoch}.pth",
        )

        os.makedirs(f"{self.logger.log_dir}/reconstructions", exist_ok=True)
        with torch.no_grad():
            recon = self.model(
                self.trainer.datamodule.data[1].unsqueeze(0).float().to("cuda")
            )
        torch.save(
            recon,
            f"{self.logger.log_dir}/reconstructions/{self.current_epoch}.pth",
        )

    def on_fit_end(self):
        """Saves the trained model at the end of training."""
        os.makedirs(f"{self.logger.log_dir}", exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.logger.log_dir}/model.pth")
        print("Model saved")
