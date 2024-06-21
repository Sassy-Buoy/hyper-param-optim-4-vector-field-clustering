""" Hyperparameter optimization with Optuna """

from sklearn.model_selection import train_test_split
import torch
import optuna

from load_data import sim_arr
from search_space import search_space
from vae import Encoder, Decoder, VarAutoEncoder
from cross_validation import cross_val


# reshape from batch, height, width, channel, to batch, channel, height, width
sim_arr_transformed = sim_arr.reshape(
    sim_arr.shape[0], sim_arr.shape[3], sim_arr.shape[1], sim_arr.shape[2])
sim_arr_tensor = torch.tensor(sim_arr_transformed, dtype=torch.float32)
# train test split
train_data, test_data = train_test_split(
    sim_arr_tensor, test_size=0.2, random_state=42, shuffle=True)


def objective(trial):
    """Objective function for hyperparameter optimization."""

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)

    # search space
    num_layers, poolsize, channels, kernel_sizes, dilations, activations = search_space(
        trial, 3, 10)

    # define model
    encoder = Encoder(num_layers, poolsize, channels,
                      kernel_sizes, dilations, activations)
    decoder = Decoder(encoder)
    model = VarAutoEncoder(encoder, decoder)

    # train model with k-fold cross validation
    val_losses = cross_val(model, train_data, n_splits=5,
                           device='cuda', lr=lr, epochs=100, batch_size=32)
    loss = sum(val_losses) / len(val_losses)

    return loss


study = optuna.create_study(direction='minimize',
                            pruner=optuna.pruners.HyperbandPruner(),
                            study_name='vae_us',
                            storage='sqlite:///vae.db',
                            load_if_exists=True)

study.optimize(objective, n_trials=20)