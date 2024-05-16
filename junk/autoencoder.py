from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Conv2DTranspose, Flatten
from tensorflow.keras.models import Sequential


class Autoencoder(Sequential):
    """Autoencoder class that inherits from Keras' Sequential class."""
    def __init__(self,
                 num_layers,
                 poolsize,
                 filters,
                 kernel_sizes,
                 strides,
                 activations,
                 paddings,
                 kernel_initializers,
                 output_dim,
                 input_dim=(80, 80, 3)):
        super().__init__()

        # set output_dim
        filters.append(output_dim)

        # Encoder
        self.encoder = Sequential(name='encoder')
        self.encoder.add(InputLayer(input_shape=input_dim))
        for i in range(num_layers):
            self.encoder.add(MaxPool2D(poolsize[num_layers - 1][i]))
            self.encoder.add(Conv2D(
                filters[i],
                kernel_sizes[i],
                strides=strides[i],
                activation=activations[i],
                padding=paddings[i],
                kernel_initializer=kernel_initializers[i]))
        self.encoder.add(MaxPool2D(5))

        # Decoder
        self.decoder = Sequential(name='decoder')
        self.decoder.add(InputLayer(input_shape=self.encoder.output_shape[1:]))
        for i in range(num_layers - 1, -1, -1):
            self.decoder.add(Conv2DTranspose(
                filters[i],
                kernel_sizes[i],
                strides=poolsize[num_layers - 1][i],
                activation=activations[i],
                padding='same',
                kernel_initializer=kernel_initializers[i]
            ))
        self.decoder.add(Conv2DTranspose(
            input_dim[-1],
            kernel_sizes[0],
            strides=5,
            activation=activations[0],
            padding='same',
            kernel_initializer=kernel_initializers[0]))

        # Compile
        self.add(self.encoder)
        self.add(self.decoder)
        self.compile(optimizer='adam', loss='mse')

    def feature_extractor(self, sim_arr):
        """Extract features from the sim data"""
        feature_extractor = Sequential([self.encoder,
                                        Flatten()])

        feature_arr = feature_extractor.predict(sim_arr, verbose=2)
        return feature_arr
