import numpy as np
import deeppy as dp

# Setup neural network
pool_kwargs = {
    'win_shape': (3, 3),
    'strides': (2, 2),
    'border_mode': 'same',
    'method': 'max',
}
net = dp.NeuralNetwork(
    layers=[
        dp.Convolutional(
            n_filters=32,
            filter_shape=(5, 5),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.0001),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.Activation('relu'),
        dp.Pool(**pool_kwargs),
        dp.Convolutional(
            n_filters=32,
            filter_shape=(5, 5),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.Activation('relu'),
        dp.Pool(**pool_kwargs),
        dp.Convolutional(
            n_filters=64,
            filter_shape=(5, 5),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.Activation('relu'),
        dp.Pool(**pool_kwargs),
        dp.Flatten(),
        dp.FullyConnected(
            n_output=64,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.Activation('relu'),
        dp.FullyConnected(
            n_output=6	,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.MeanSquaredError(),
    ],
)