import numpy as np

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