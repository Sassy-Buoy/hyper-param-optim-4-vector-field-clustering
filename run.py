"""run.py"""

import lightning as L
import yaml
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from models import LitModel, DataModule

# Define the logger
logger = CSVLogger("")

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    dirpath=logger.log_dir,  # Directory to save the checkpoints
    filename="model-{epoch:02d}-{val_loss:.3f}",  # Checkpoint filename
    save_top_k=3,  # Save only the best model
    mode="min",  # Minimize the monitored metric
    auto_insert_metric_name=False,  # Avoid automatic metric name insertion in filename
)

trainer = L.Trainer(
    precision="16-mixed",
    callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=50)],
    max_epochs=200,
    logger=logger,
    accumulate_grad_batches=2,
    log_every_n_steps=1,
)

# get model_type and config from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

litmodel = LitModel(
    model_type=config["model_type"],
    config=config["config"],
    learning_rate=config["learning_rate"],
    threshold=config["threshold"] if config["model_type"] == "variational" else None,
)

litmodel.model.load_state_dict(torch.load("lightning_logs/version_4/model.pth"))

trainer.fit(
    litmodel,
    datamodule=DataModule(batch_size=int(config["batch_size"] / 2), num_workers=4),
)