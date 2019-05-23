#!/usr/bin/env python
from keras.datasets import cifar100
from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt


def twospirals(n_points, noise=0.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points), np.ones(n_points))),
    )


X, y = twospirals(1000)
print(X)
plt.title("training set")
plt.plot(X[y == 0, 0], X[y == 0, 1], ".")
plt.plot(X[y == 1, 0], X[y == 1, 1], ".")
plt.legend()
plt.show()
