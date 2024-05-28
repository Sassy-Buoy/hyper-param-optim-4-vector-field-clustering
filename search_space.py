import torch.nn as nn


def search_space(trial, input_dim, output_dim):
    """ define the hyperparameter search space."""

    num_layers = trial.suggest_int('num_layers', 2, 5)
    # num_layers = 5

    if num_layers == 2:
        poolsize = [8, 10]
        """poolsize = trial.suggest_categorical('poolsize_2', [[5, 16], [16, 5],
                                                            [8, 10], [10, 8],
                                                            [4, 20], [20, 4],
                                                            [2, 40], [40, 2]])"""

    elif num_layers == 3:
        poolsize = trial.suggest_categorical(
            'poolsize_3', [[4, 4, 5], [4, 5, 4], [5, 4, 4]])
        """poolsize = trial.suggest_categorical(
            'poolsize_3', [[2, 2, 20], [2, 20, 2], [20, 2, 2],
                           [4, 4, 5], [4, 5, 4], [5, 4, 4],
                           [2, 5, 8], [2, 8, 5], [5, 2, 8],
                           [5, 8, 2], [8, 2, 5], [8, 5, 2],
                           [2, 4, 10], [2, 10, 4], [4, 2, 10],
                           [4, 10, 2], [10, 2, 4], [10, 4, 2]])"""

    elif num_layers == 4:
        poolsize = trial.suggest_categorical(
            'poolsize_4', [[2, 2, 4, 5], [2, 2, 5, 4], [2, 4, 2, 5],
                           [2, 4, 5, 2], [2, 5, 2, 4], [2, 5, 4, 2],
                           [4, 2, 2, 5], [4, 2, 5, 2], [4, 5, 2, 2],
                           [5, 2, 2, 4], [5, 2, 4, 2], [5, 4, 2, 2]])
        """poolsize = trial.suggest_categorical(
            'poolsize_4', [[2, 2, 2, 10], [2, 2, 10, 2], [2, 10, 2, 2], [10, 2, 2, 2],
                           [2, 2, 4, 5], [2, 2, 5, 4], [2, 4, 2, 5],
                           [2, 4, 5, 2], [2, 5, 2, 4], [2, 5, 4, 2],
                           [4, 2, 2, 5], [4, 2, 5, 2], [4, 5, 2, 2],
                           [5, 2, 2, 4], [5, 2, 4, 2], [5, 4, 2, 2]])"""

    elif num_layers == 5:
        poolsize = trial.suggest_categorical(
            'poolsize_5', [[2, 2, 2, 2, 5], [2, 2, 2, 5, 2], [2, 2, 5, 2, 2],
                           [2, 5, 2, 2, 2], [5, 2, 2, 2, 2]])

    # poolsize = [2, 2, 2, 2, 5]

    channels = [input_dim,]
    for i in range(num_layers - 1):
        channels.append(trial.suggest_int(
            f'channels_{i}', 2, 16))
    channels.append(output_dim)

    # channels = [3, 11, 10, 12, 10, 3]

    kernel_sizes = [trial.suggest_int(
        f'kernel_size_{i}', 2, 16) for i in range(num_layers)]

    dilations = [trial.suggest_int(
        f'dilation_{i}', 1, 5) for i in range(num_layers)]

    activations = [trial.suggest_categorical(
        f'activation_{i}', ['nn.Softplus',
                            'nn.SELU',
                            'nn.SiLU',
                            'nn.Tanh']) for i in range(num_layers)]

    activations = [eval(activation) for activation in activations]

    return num_layers, poolsize, channels, kernel_sizes, dilations, activations
