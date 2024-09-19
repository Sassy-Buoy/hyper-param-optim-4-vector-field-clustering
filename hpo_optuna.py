"""run"""
from torch import nn
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
import optuna
from models import LitAE


# load data from data folder(we"re in notebooks folder)
sim_arr_tensor = torch.load('./data/sim_arr_tensor.pt')

# train test split
train_data, test_data = train_test_split(
    sim_arr_tensor, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42)


def objective(trial):
    """ define the objective function."""
    num_layers = 5
    channels = [3,]
    for i in range(num_layers - 1):
        channels.append(trial.suggest_int(
            f'channels_{i}', 7, 20))
    kernel_sizes = [trial.suggest_categorical(
        f'kernel_{i}', [2, 4, 8, 16, 32, 64]) for i in range(num_layers)]

    dilations = [trial.suggest_int(
        f'dilation_{i}', 1, 5) for i in range(num_layers)]

    activations = [trial.suggest_categorical(
        f'activation_{i}', ['nn.Softplus',
                            'nn.SELU',
                            'nn.SiLU',
                            'nn.Tanh']) for i in range(num_layers)]

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    activations = [eval(activation) for activation in activations]
    channels.append(12)
    hyperparameters_dict = {
        "num_layers": 5,
        "poolsize": [2, 2, 2, 2, 5],
        "channels": channels,
        "kernel_sizes": kernel_sizes,
        "dilations": dilations,
        "activations": activations,
        "lr": lr,
        "batch_size": batch_size
    }

    lit_model = LitAE(hyperparameters=hyperparameters_dict)
    trainer = L.Trainer(callbacks=[EarlyStopping(
        monitor="val_loss")], max_epochs=100, accelerator="cuda")
    trainer.fit(model=lit_model,
                train_dataloaders=DataLoader(train_data,
                                             batch_size=hyperparameters_dict["batch_size"],
                                             num_workers=31),
                val_dataloaders=DataLoader(val_data,
                                           batch_size=hyperparameters_dict["batch_size"],
                                           num_workers=31))

    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss


study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.HyperbandPruner(),
                            study_name='AutoEncoder_10',
                            storage='sqlite:///optuna.db',
                            load_if_exists=True)
study.optimize(objective, n_trials=100)
