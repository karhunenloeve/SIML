import numpy as np
import math
import scipy

def gaussian_curvature(data: np.ndarray) -> np.ndarray:
    """
    Computes the gaussian curvature of a numpy ndarray.
    :param data: A numpy ndarray.
    :return: Gaussian curvature.
    """
    Zy, Zx = np.gradient(data)
    Zxy, Zxx = np.gradient(Zx)
    Zyy, _ = np.gradient(Zy)
    K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2

    return K

def mean_curvature(data: np.ndarray) -> np.ndarray:
    """
    Computes the mean curvature of a numpy ndarray.
    :param data: Points ensembled from the data manifold.
    :return: Mean curvature scalar.
    """
    Zy, Zx  = numpy.gradient(data)
    Zxy, Zxx = numpy.gradient(Zx)
    Zyy, _ = numpy.gradient(Zy)

    H = (Zx**2 + 1)*Zyy - 2*Zx*Zy*Zxy + (Zy**2 + 1)*Zxx
    H = -H/(2*(Zx**2 + Zy**2 + 1)**(1.5))

    return H

def distcorr(X, Y):
    """
    Compute the distance correlation function on two random variables X and Y.
    :param X: Attribute column X.
    :param Y: Attribute column Y.
    :return: Distance correlation.
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]

    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')

    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor

