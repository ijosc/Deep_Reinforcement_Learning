import numpy as np
import deeppy as dp

# Setup neural network
net = dp.NeuralNetwork(
    layers=[
        dp.Convolutional(
            n_filters=16,
            filter_shape=(8, 8),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.0001),
                                 weight_decay=0.004, monitor=True),
            strides=(4,4),
        ),
        dp.Activation('relu'),
        dp.Convolutional(
            n_filters=32,
            filter_shape=(4, 4),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                 weight_decay=0.004, monitor=True),
            strides=(2,2),
        ),
        dp.Activation('relu'),
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
        dp.MeanSquaredError(),
    ],
)