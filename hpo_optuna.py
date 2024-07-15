""" Hyperparameter optimization with Optuna """

from sklearn.model_selection import train_test_split
import torch
import optuna

from load_data import sim_arr
from search_space import search_space
from vae import Encoder, Decoder, VaDE  # VarAutoEncoder
from cross_validation import cross_val

# reshape from batch, height, width, channel, to batch, channel, height, width
sim_arr_transformed = sim_arr.reshape(
    (sim_arr.shape[0], sim_arr.shape[3], sim_arr.shape[1], sim_arr.shape[2]))
sim_arr_tensor = torch.tensor(sim_arr_transformed, dtype=torch.float32)
# train test split
train_data, test_data = train_test_split(
    sim_arr_tensor, test_size=0.2, random_state=42, shuffle=True)


def objective(trial):
    """Objective function for hyperparameter optimization."""

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 32, 64, 96, 128)
    epochs = trial.suggest_int('epochs', 50, 100, 150, 200)

    # search space
    num_layers, poolsize, channels, kernel_sizes, dilations, activations = search_space(
        trial, input_dim=3, output_dim=3)

    # define model
    encoder_ = Encoder(num_layers, poolsize, channels,
                       kernel_sizes, dilations, activations)
    decoder_ = Decoder(encoder_)
    model_ = VaDE(encoder_, decoder_, 30)

    # train model with k-fold cross validation
    val_losses_ = cross_val(model_, train_data, lr=lr, batch_size=batch_size,
                            epochs=epochs, n_splits=5, device='cuda')
    loss_ = sum(val_losses_) / len(val_losses_)

    return loss_


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize',
                                pruner=optuna.pruners.HyperbandPruner(),
                                study_name='vade',
                                storage='sqlite:///optuna.db',
                                load_if_exists=True)

    study.optimize(objective, n_trials=1)

    best_trial = study.best_trial

    encoder = Encoder(*search_space(best_trial, 3, 10))
    decoder = Decoder(encoder)
    model = VaDE(encoder, decoder, 30)

    # train model with k-fold cross validation
    val_losses = cross_val(model, train_data,
                           lr=best_trial.params['lr'],
                           batch_size=best_trial.params['batch_size'],
                           epochs=best_trial.params['epochs'],
                           n_splits=5, device='cuda')
    loss = sum(val_losses) / len(val_losses)

    # save best model
    torch.save(model.state_dict(), 'best_vade.pth')
