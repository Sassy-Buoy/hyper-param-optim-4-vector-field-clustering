from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf
import optuna
from sklearn.model_selection import train_test_split
import pathlib as pl
import discretisedfield as df
import numpy as np
import json

simulation_file_paths = list(
    pl.Path("sims").glob("Configs_*/drive-[0-9]/Configs_*.omf")
)

sim_arr = np.array([df.Field.from_file(file).orientation.sel(
    "z").array for file in simulation_file_paths])

parameters_dict = {}
for path in simulation_file_paths:
    json_file_path = path.parent / "parameters_DE.json"
    with open(json_file_path, "r", encoding="utf-8") as f_handle:
        parameters_dict[str(path)] = json.load(f_handle)


train_set, test_set = train_test_split(sim_arr, test_size=0.2, random_state=42)
train_set, valid_set = train_test_split(
    train_set, test_size=0.2, random_state=42)

print(train_set.shape[0], valid_set.shape[0], test_set.shape[0])


# Define the input shape
input_shape = (80, 80, 3)


def create_model(trial):
    # Define the parameters to search over
    num_layers = trial.suggest_int('num_layers', 1, 4)
    num_filters = [trial.suggest_categorical(
        f'num_filters_{i}', [3, 6, 9, 12]) for i in range(num_layers)]
    kernel_size = trial.suggest_categorical('kernel_size', [4, 8])
    activation = trial.suggest_categorical('activation', ['relu', 'selu'])

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Encoder
    x = inputs
    for i in range(num_layers):
        x = Conv2D(num_filters[i], kernel_size,
                   activation=activation, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Decoder
    for i in range(num_layers - 1, -1, -1):
        x = Conv2DTranspose(num_filters[i], kernel_size, strides=(
            2, 2), activation=activation, padding='same')(x)

    # Output layer
    outputs = Conv2D(input_shape[-1], kernel_size,
                     activation='sigmoid', padding='same')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model


def objective(trial):
    # Create model
    model = create_model(trial)

    # Compile model
    model.compile(optimizer=Adam(), loss='mse')

    # Train the model
    history = model.fit(train_set, train_set, batch_size=64, epochs=10,
                        validation_data=(valid_set, valid_set))

    # Return validation loss
    return history.history['val_loss'][-1]


# Run Optuna optimization
study = optuna.create_study(direction='minimize', study_name='autoencoder',
                            storage='sqlite:///autoencoder.db', load_if_exists=True)
study.optimize(objective, n_trials=100)

# Print best hyperparameters
print('Best trial:')
trial = study.best_trial
print(f'  Loss: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
