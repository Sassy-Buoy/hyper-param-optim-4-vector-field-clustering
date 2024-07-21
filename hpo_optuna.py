""" Hyperparameter optimization with Optuna """

from sklearn.model_selection import train_test_split
import torch
import optuna

from load_data import sim_arr
from search_space import search_space
from ae import Encoder, Decoder, AutoEncoder
from cross_validation import train_model, evaluate_model

# reshape from batch, height, width, channel, to batch, channel, height, width
sim_arr_transformed = sim_arr.reshape(
    (sim_arr.shape[0], sim_arr.shape[3], sim_arr.shape[1], sim_arr.shape[2]))
sim_arr_tensor = torch.tensor(sim_arr_transformed, dtype=torch.float32)
# train test split
train_data, test_data = train_test_split(
    sim_arr_tensor, test_size=0.2, random_state=42, shuffle=True)


def objective(trial):
    """Objective function for hyperparameter optimization."""

    # clear cuda cache
    torch.cuda.empty_cache()

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
    epochs = trial.suggest_categorical('epochs', [50, 100, 150, 200])

    # search space
    num_layers, poolsize, channels, kernel_sizes, dilations, activations = search_space(
        trial, input_dim=3, output_dim=12)

    # define model
    encoder_ = Encoder(num_layers, poolsize, channels,
                       kernel_sizes, dilations, activations)
    decoder_ = Decoder(encoder_)
    model_ = AutoEncoder(encoder_, decoder_)

    # train model with k-fold cross validation
    train_model(model_, train_data,
                lr=lr, batch_size=batch_size, epochs=epochs)
    loss_ = evaluate_model(model_, test_data, batch_size=batch_size)

    return loss_


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize',
                                pruner=optuna.pruners.HyperbandPruner(),
                                study_name='ae_12',
                                storage='sqlite:///optuna.db',
                                load_if_exists=True)

    study.optimize(objective, n_trials=13)
