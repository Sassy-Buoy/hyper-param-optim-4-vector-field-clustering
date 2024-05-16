""" Hyperparameter optimization for Autoencoder"""
import numpy as np
import tensorflow as tf
import optuna
from sklearn.model_selection import KFold


class Hyper:
    """Hyperparameter optimization class"""

    def __init__(self, model, output_dim, train_set, test_set, n_trials=0, n_splits=5):
        self.model = model
        self.output_dim = output_dim
        self.train_set = train_set
        self.train_set = test_set
        self.n_splits = n_splits

    def search_space(self, trial):
        """ Define the search space for hyperparameters"""
        num_layers = trial.suggest_int('num_layers', 1, 10)

        poolsize = [[16],
                    [4, 4],
                    [2, 2, 4],
                    [2, 2, 2, 2]]

        filters = [trial.suggest_int(f'filters_{i}', 1, 12)
                   for i in range(num_layers - 1)]

        kernel_sizes = [trial.suggest_int(
            f'kernel_size{i}', 2, 24) for i in range(num_layers)]

        strides = [trial.suggest_int(
            f'strides_{i}', 1, 3) for i in range(num_layers)]

        activations = [trial.suggest_categorical(
            f'activation_{i}', ['relu', 'selu', 'tanh', 'sigmoid']) for i in range(num_layers)]

        paddings = ['same' for i in range(num_layers)]
        #paddings = [trial.suggest_categorical(
        #    f'padding_{i}', ['same', 'same']) for i in range(num_layers)]

        kernel_initializers = [(trial.suggest_categorical(
            f'initializer_{i}', ['lecun_normal',
                                 'he_normal',
                                 'he_uniform',
                                 'glorot_normal',
                                 'glorot_uniform'])) for i in range(num_layers)]

        return [num_layers, poolsize, filters, kernel_sizes,
                strides, activations, paddings, kernel_initializers]

    def objective(self, trial):
        """ Objective function for the hyperparameter optimization"""
        # clear the previous session
        tf.keras.backend.clear_session()

        model = self.model(*self.search_space(trial),
                           output_dim=self.output_dim)

        # Train the model
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        val_losses = []
        for train_index, val_index in kf.split(self.train_set):
            train, val = self.train_set[train_index], self.train_set[val_index]
            history = model.fit(train, train, epochs=10, batch_size=32,
                                validation_data=(val, val))
        val_losses.append(history.history['val_loss'][-1])

        return np.mean(val_losses)

    def optimize(self, direction, study_name, storage, n_trials=0):
        """ Optimize the hyperparameters"""
        study = optuna.create_study(direction=direction,
                                    pruner=optuna.pruners.MedianPruner(),
                                    study_name=study_name,
                                    storage='sqlite:///' + storage,
                                    load_if_exists=True,)

        study.optimize(self.objective,
                       n_trials=n_trials,
                       show_progress_bar=True)

        return study
