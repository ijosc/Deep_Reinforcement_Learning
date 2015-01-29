import numpy as np
import deeppy as dp

net = dp.NeuralNetwork(
    layers=[
        dp.Convolutional(
            n_filters=32,
            filter_shape=(5, 5),
            weights=dp.Parameter(dp.AutoFiller(), weight_decay=0.0001),
        ),
        dp.Activation('relu'),
        dp.Pool(
            win_shape=(3, 3),
            strides=(2, 2),
            method='max',
        ),
        dp.Convolutional(
            n_filters=64,
            filter_shape=(5, 5),
            weights=dp.Parameter(dp.AutoFiller(), weight_decay=0.0001),
        ),
        dp.Activation('relu'),
        dp.Pool(
            win_shape=(3, 3),
            strides=(2, 2),
            method='max',
        ),
        dp.Flatten(),
        dp.FullyConnected(
            n_output=128,
            weights=dp.Parameter(dp.AutoFiller()),
        ),
        dp.FullyConnected(
            n_output=6,
            weights=dp.Parameter(dp.AutoFiller()),
        ),
        dp.MultinomialLogReg(),
    ],
)