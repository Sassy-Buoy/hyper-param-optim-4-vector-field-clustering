"""run"""
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import nn
import torch
from sklearn.model_selection import train_test_split
import optuna
from models import LitVaDE
from cluster_acc import adj_rand_index

# load data from data folder(we"re in notebooks folder)
sim_arr_tensor = torch.load('./data/sim_arr_tensor.pt')

# train test split
train_data, test_data = train_test_split(
    sim_arr_tensor, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42)


def objective(trial):
    hyperparameters_dict = {
        "num_layers": 5,
        "poolsize": [2, 2, 2, 2, 5],
        "channels": [3, 12, 7, 7, 11, 12],
        "kernel_sizes": [15, 10, 14, 14, 14],
        "dilations": [5, 5, 3, 3, 1],
        "activations": [nn.SiLU, nn.Tanh, nn.Softplus, nn.SiLU, nn.SiLU],
        "lr": 0.0002568457560593339,
        "batch_size": 32,
        "beta": trial.suggest_float('beta', 1, 1000, log=True),
        "n_clusters": 13
    }

    lit_model = LitVaDE(hyperparameters=hyperparameters_dict)

    # load pretrained weights for encoder and decoder
    lit_model.model.encoder.load_state_dict(torch.load(
        './saved_models/encoder_weights.pt'), strict=False)
    lit_model.model.decoder.load_state_dict(
        torch.load('./saved_models/decoder_weights.pt'))

    trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss")])
    trainer.fit(model=lit_model,
                train_dataloaders=DataLoader(train_data,
                                             batch_size=hyperparameters_dict["batch_size"],
                                             num_workers=31),
                val_dataloaders=DataLoader(val_data,
                                           batch_size=hyperparameters_dict["batch_size"],
                                           num_workers=31))

    labels = lit_model.model.classify(sim_arr_tensor)
    labels = labels.cpu().detach().numpy()

    return adj_rand_index(labels)


study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.HyperbandPruner(),
                            study_name='run',
                            storage='sqlite:///vae_study.db',
                            load_if_exists=True)
study.optimize(objective, n_trials=13)
