import numpy as np
import deeppy as dp

# Setup neural network
pool_kwargs1 = {
    'win_shape': (8, 8),
    'strides': (4, 4),
    'border_mode': 'same',
    'method': 'max',
}
pool_kwargs2 = {
    'win_shape': (4, 4),
    'strides': (2, 2),
    'border_mode': 'same',
    'method': 'max',
}
net = dp.NeuralNetwork(
    layers=[
        dp.Convolutional(
            n_filters=16,
            filter_shape=(8, 8),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.0001),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.Activation('relu'),
        dp.Pool(**pool_kwargs1),
        dp.Convolutional(
            n_filters=32,
            filter_shape=(4, 4),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.Activation('relu'),
        dp.Pool(**pool_kwargs2),
        dp.Flatten(),
        dp.FullyConnected(
            n_output=256,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.Activation('relu'),
        dp.FullyConnected(
            n_output=6	,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                 weight_decay=0.004, monitor=True),
        ),
        dp.MultinomialLogReg(),
    ],
)