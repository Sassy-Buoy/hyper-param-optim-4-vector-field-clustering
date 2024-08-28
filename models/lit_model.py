"""Lightning module for training and evaluation."""

import torch
import lightning as L


class LitModel(L.LightningModule):
    """Lightning module for training and evaluation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # check if model is vanilla or variational
        if hasattr(self.model, "get_loss"):
            loss = self.model.get_loss(batch)
            self.log("train_loss", loss)
        else:
            reconstruction_loss = self.model.get_reconstruction_loss(batch)
            kl_divergence = self.model.get_kl_divergence(batch)
            self.log("train_reconstruction_loss", reconstruction_loss)
            self.log("train_kl_divergence", kl_divergence)
            loss = reconstruction_loss + kl_divergence
        return loss

    def validation_step(self, batch, batch_idx):
        # check if model is vanilla or variational
        if hasattr(self.model, "get_loss"):
            loss = self.model.get_loss(batch)
            self.log("val_loss", loss)
        else:
            reconstruction_loss = self.model.get_reconstruction_loss(batch)
            kl_divergence = self.model.get_kl_divergence(batch)
            self.log("val_reconstruction_loss", reconstruction_loss)
            self.log("val_kl_divergence", kl_divergence)
            loss = reconstruction_loss + kl_divergence
        return loss

    def test_step(self, batch, batch_idx):
        # check if model is vanilla or variational
        if hasattr(self.model, "get_loss"):
            loss = self.model.get_loss(batch)
            self.log("test_loss", loss)
        else:
            reconstruction_loss = self.model.get_reconstruction_loss(batch)
            kl_divergence = self.model.get_kl_divergence(batch)
            self.log("test_reconstruction_loss", reconstruction_loss)
            self.log("test_kl_divergence", kl_divergence)
            loss = reconstruction_loss + kl_divergence
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
