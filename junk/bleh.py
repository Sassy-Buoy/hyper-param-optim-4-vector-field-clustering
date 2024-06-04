""" Hyperparameter tuning with Ray Tune and Optuna. 
"""

import torch
from sklearn.model_selection import train_test_split

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

from vae import Encoder, Decoder, VarAutoEncoder
from cross_validation import cross_val
from load_data import sim_arr


# reshape from batch, height, width, channel, to batch, channel, height, width
sim_arr_transformed = sim_arr.reshape(sim_arr.shape[0],
                                      sim_arr.shape[3],
                                      sim_arr.shape[1],
                                      sim_arr.shape[2])

sim_arr_tensor = torch.tensor(sim_arr_transformed, dtype=torch.float32)


# train test split
train_data, test_data = train_test_split(
    sim_arr_tensor, test_size=0.2, random_state=42, shuffle=True)

train_data_ref = ray.put(train_data)


def get_config():
    """ Define the hyperparameter search space."""

    config = {'num_layers': tune.randint(2, 5)}

    if config['num_layers'] == 2:
        config['poolsize'] = [8, 10]
    elif config['num_layers'] == 3:
        config['poolsize'] = tune.choice([[4, 4, 5], [4, 5, 4], [5, 4, 4]])
    elif config['num_layers'] == 4:
        config['poolsize'] = tune.choice([[2, 2, 4, 5], [2, 2, 5, 4], [2, 4, 2, 5],
                                          [2, 4, 5, 2], [2, 5, 2, 4], [2, 5, 4, 2],
                                          [4, 2, 2, 5], [4, 2, 5, 2], [4, 5, 2, 2],
                                          [5, 2, 2, 4], [5, 2, 4, 2], [5, 4, 2, 2]])
    elif config['num_layers'] == 5:
        config['poolsize'] = tune.choice([[2, 2, 2, 2, 5], [2, 2, 2, 5, 2], [2, 2, 5, 2, 2],
                                          [2, 5, 2, 2, 2], [5, 2, 2, 2, 2]])

    for i in range(5):
        config[f'channels_{i}'] = tune.randint(2, 17)
        config[f'kernel_size_{i}'] = tune.randint(2, 17)
        config[f'dilation_{i}'] = tune.randint(1, 6)
        config[f'activation_{i}'] = tune.choice(
            ['nn.Softplus', 'nn.SELU', 'nn.SiLU', 'nn.Tanh'])

    config['lr'] = tune.loguniform(1e-5, 1e-3)

    return config


def get_params(config, output_dim, input_dim=3):
    """Extract the hyperparameters from the config."""

    num_layers = config['num_layers']

    channels = [input_dim]
    for i in range(num_layers - 1):
        channels.append(config[f'channels_{i}'])
    channels.append(output_dim)

    poolsize = config['poolsize']

    kernel_sizes = [config[f'kernel_size_{i}'] for i in range(num_layers)]

    dilations = [config[f'dilation_{i}'] for i in range(num_layers)]

    activations = [config[f'activation_{i}'] for i in range(num_layers)]
    activations = [eval(activation) for activation in activations]

    lr = config['lr']

    return num_layers, poolsize, channels, kernel_sizes, dilations, activations, lr


def train_vae(config):
    """ Train the Variational Autoencoder with the given hyperparameters.
    """

    train_data = ray.get(train_data_ref)

    num_layers, poolsize, channels, kernel_sizes, dilations, activations, lr = get_params(
        config, 10)

    # define model
    encoder = Encoder(num_layers, poolsize, channels,
                      kernel_sizes, dilations, activations)
    decoder = Decoder(encoder)
    model = VarAutoEncoder(encoder, decoder)

    # train model with k-fold cross validation
    val_losses = cross_val(model, train_data, n_splits=5,
                           epochs=100, batch_size=32, lr=lr)

    val_loss = sum(val_losses) / len(val_losses)

    tune.report(loss=val_loss)


if __name__ == '__main__':
    # Initialize Ray
    ray.init(address='auto', ignore_reinit_error=True)

    # num cpu cores and gpus available
    print('Number of CPU cores:', ray.available_resources()['CPU'])
    print('Number of GPUs:', ray.available_resources()['GPU'])

    # Register the train_data object
    train_data_ref = ray.put(train_data)

    analysis = tune.run(
        train_vae,
        config=get_config(),
        metric='loss',
        mode='min',
        search_alg=OptunaSearch(),
        scheduler=ASHAScheduler(),
        num_samples=10,
        resources_per_trial={'cpu': 4, 'gpu': 1},
        verbose=1,
        local_dir='~/ray_results',
        # checkpoint_at_end=True,
        # checkpoint_freq=1,
        # max_failures=3,
        # stop={'training_iteration': 100}
    )

    print('Best hyperparameters found were: ', analysis.best_config)
    print('Best loss found was: ', analysis.best_result['loss'])
    print('Best trial found was: ', analysis.best_trial)
    print('Best trial config found was: ', analysis.best_trial.config)
    print('Best trial result found was: ', analysis.best_trial.result)
