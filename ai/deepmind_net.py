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
                                 weight_decay=0.004, monitor=False),
            strides=(4,4),
        ),
        dp.Activation('relu'),
        dp.Convolutional(
            n_filters=32,
            filter_shape=(4, 4),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                 weight_decay=0.004, monitor=False),
            strides=(2,2),
        ),
        dp.Activation('relu'),
        dp.Convolutional(
            n_filters=64,
            filter_shape=(3, 3),
            border_mode='same',
            weights=dp.Parameter(dp.NormalFiller(sigma=0.01),
                                 weight_decay=0.004, monitor=False),
            strides=(1,1),
        ),
        dp.Activation('relu'),
        dp.Flatten(),
        dp.FullyConnected(
            n_output=512,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                 weight_decay=0.004, monitor=False),
        ),
        dp.Activation('relu'),
        dp.FullyConnected(
            n_output=1,
            weights=dp.Parameter(dp.NormalFiller(sigma=0.1),
                                 weight_decay=0.004, monitor=False),
        ),
        dp.MeanSquaredError(),
    ],
)