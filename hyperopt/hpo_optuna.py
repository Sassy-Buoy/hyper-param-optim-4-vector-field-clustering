""" Hyperparameter optimization with Optuna """

import torch
from sklearn.model_selection import train_test_split

# load data from data folder(we"re in notebooks folder)
sim_arr_tensor = torch.load('./data/sim_arr_tensor.pt')

# train test split
train_data, test_data = train_test_split(
    sim_arr_tensor, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42)

from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import optuna

from models import variational, LitModel
from cluster_acc import adj_rand_index
from search_space import search_space

def objective(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128, 256])

    num_layers, poolsize, channels, kernel_sizes, dilations, activations = search_space(
        trial, input_dim=3, output_dim=12)
    
    beta = trial.suggest_float('beta', 1, 1000, log=True)
    

    encoder = variational.Encoder(num_layers, poolsize, channels,
                            kernel_sizes, dilations, activations)
    decoder = variational.Decoder(encoder)
    model = variational.DeepEmbedding(encoder, decoder, n_clusters=13)

    lit_model = LitModel(model, lr=lr, beta=beta)

    trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss")])
    trainer.fit(model=lit_model,
                train_dataloaders=DataLoader(train_data, batch_size=batch_size, num_workers=31),
                val_dataloaders=DataLoader(val_data, batch_size=batch_size, num_workers=31))

    labels = model.classify(sim_arr_tensor)
    labels = labels.cpu().detach().numpy()

    return adj_rand_index(labels)
study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.HyperbandPruner(),
                            study_name='vae_study',
                            storage='sqlite:///vae_study.db',
                            load_if_exists=True)
study.optimize(objective, n_trials=100)


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize',
                                pruner=optuna.pruners.HyperbandPruner(),
                                study_name='ae_12',
                                storage='sqlite:///optuna.db',
                                load_if_exists=True)

    study.optimize(objective, n_trials=13)
