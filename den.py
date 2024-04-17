# Define Clustering Layer
class ClusteringLayer(tf.keras.layers.Layer):
    def __init__(self, num_clusters, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.num_clusters = num_clusters

    def build(self, input_shape):
        self.clusters = self.add_weight(name='clusters',
                                        shape=(self.num_clusters,
                                               input_shape[1]),
                                        initializer='glorot_uniform',
                                        trainable=True)

    def call(self, inputs, **kwargs):
        q = 1.0 / \
            (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)))
        q **= (1.0 + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

# Build Deep Embedding Network


def build_DEN(num_clusters, beta, auto_encoder=auto_encoder):
    clustering_layer = ClusteringLayer(num_clusters)
    inputs = Input(shape=(80, 80, 3))
    encoder = Sequential([
        auto_encoder.get_layer("encoder"),
        Flatten(),])
    encoded = encoder(inputs)
    q_values = clustering_layer(encoded)
    decoder = Sequential([ Reshape((1, 1, 2)),
        auto_encoder.get_layer("decoder"),
        ])
    decoded = decoder(encoded)
    den_model = Model(inputs=inputs, outputs=[decoded, q_values])
    den_model.compile(optimizer="adam",
                      loss=['mse', 'kld'],
                      loss_weights=[1-beta, beta])
    return den_model

# validation split
train_set, valid_set = train_test_split(train_set, test_size=0.2, random_state=42)

def objective(trial):
    # Clear clutter from previous sessions
    tf.keras.backend.clear_session()

    num_clusters = trial.suggest_int('num_clusters', 10, 20)
    beta = trial.suggest_float('beta', 0.1, 1)

    # Create model
    model = build_DEN(num_clusters=num_clusters, alpha=alpha, beta=beta)

    history = model.fit(train_set, train_set, epochs=10, batch_size=32,
                        validation_data=(valid_set, valid_set))
    return history.history['val_loss'][-1]

study = optuna.create_study(direction='minimize',
                            pruner=optuna.pruners.HyperbandPruner(),
                            study_name='den_cv',
                            storage='sqlite:///test.db',
                            load_if_exists=True)
study.optimize(objective, n_trials=100)


# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)
print("Best value:", study.best_value)

# Train the model with the best hyperparameters
alpha = study.best_params['alpha']
beta = study.best_params['beta']
num_clusters = study.best_params['num_clusters']
den_model = build_DEN(num_clusters, alpha, beta)
den_model.fit(train_set, train_set, epochs=100, batch_size=32)
den_model.evaluate(test_set, test_set)