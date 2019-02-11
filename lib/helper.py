#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import typing

from sklearn.utils import shuffle
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import FastICA, PCA
from scipy import stats, signal

def findCounts(arr: np.ndarray) -> np.ndarray:
    """
    Replaces the elements of the array by their relative frequencies (only 2D-arrays).
    :param arr: array of the dataset as numpy ndarray.
    :return: numpy ndarray.
    """
    array_shape = arr.shape
    # check if array is empty, if it is, return 0
    # this is the trivial case
    if len(arr) == 0:
        return arr

    # check if the array is more then two dimensional
    # if it is, then raise an error
    if len(array_shape) > 2:
        print("Dimension of ndarray must be exactly two.")
        return False
    else:
        # init the output list
        frequencies = np.empty(shape = array_shape, dtype = float)
        # calculate the statistics for each row
        for i in range(0, array_shape[1]):
            res = stats.relfreq(arr[:,i], len(arr[:,i]))
            print(res.frequency)
            np.append(frequencies, res.frequency)
    #returns numpy array with frequencies
    return frequencies


def read_data_csv(path: str, sep: str = ",") -> np.ndarray:
    """
    Creates a numpy ndarray from .csv data.
    :param path: path to the csv.
    :return: np.ndarray.
    """
    return np.genfromtxt(path, delimiter = sep)


def standardize(X: np.ndarray) -> np.ndarray:
    """
    Centering the data by subtracting the mean and dividing by standard deviation.
    :param X: An ndarray with the data.
    :return: np.ndarray.
    """
    # Standardize data along the first axis of the tuples
    X /= X.std(axis=0)
    return X


def compute_ica(S: np.ndarray, n: int) -> dict:
    """
    Computes the FastICA implementation.
    :param S: np.ndarray with data values.
    :param n: Number of independent components to be found.
    :return: {"Signals": S_, "MixingMatrix": A_, "UnmixingMatrix": U_}
    """
    ica = FastICA(n_components = n)
    S_ = ica.fit_transform(S)  # Reconstruct signals / computes ICA
    A_ = ica.mixing_  # Get estimated mixing matrix
    U_ = ica.components_  # Get the unmixing matrix

    # We can `prove` that the ICA model applies by reverting the unmixing.
    assert np.allclose(S, np.dot(S_, A_.T) + ica.mean_)
    return {"Data": S, "Signals": S_, "MixingMatrix": A_, "Components": U_}


def compute_pca(X: np.ndarray, n: int) -> dict:
    """
    Computes the principal components of the dataset.
    :param X: np.ndarray with data values.
    :param n: Number of independent components to be found.
    :return: {"Data": X, "Components": C.components_, "SingularValues": C.singular_values_}
    """
    pca = PCA(n_components = n)
    X_ = pca.fit_transform(X)
    C = pca.components_
    D = pca.singular_values_
    return {"Data": X, "Signals": X_, "MixingMatrix": C, "SingularValues": D}


def plot_ica(path: str, n: int):
    """
    Procedure plots the independent components of some data.
    :param path: string to the data, that should be plotted.
    :param n: number of components to be found.
    """
    try:
        X = read_data_csv(path) # reads the path into an np.ndarray
        s_X = standardize(X) # standardizes the data
        ica_dict = compute_ica(s_X, n) # gets a dict with Data, Signals, MixingMatrix and UnmixingMatrix
        pca_dict = compute_pca(s_X, n) # gets a dict with Data, Components and SingularValues

        # create the plotting object
        plt.figure()
        models = [] # here is specified what should be plotted

        models = [ica_dict["Components"], ica_dict["MixingMatrix"], ica_dict["Signals"], pca_dict["Signals"], X]
        names = ['Components',
                 'Mixing Matrix',
                 'ICA recovered signals',
                 'PCA recovered signals',
                 'Original Signals']
        colors = ['red', 'steelblue', 'orange', 'gray', 'green', 'yellow', 'blue', 'purple']

        for ii, (model, name) in enumerate(zip(models, names), 1):
            plt.subplot(5, 1, ii)
            plt.title(name)
            for sig, color in zip(model.T, colors):
                plt.plot(sig, color=color)

        # adjust the plots to have a nice look
        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
        plt.show()
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")


def check_linear_dependence(matrix: np.ndarray) -> boolean:
    """
    Functions checks by Cauchy-Schwartz inqeuality whether two matrices are linear dependent or not.
    :param matrix: 2x2 matrix to be processed.
    :return: Boolean.
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i != j:
                inner_product = np.inner(
                    matrix[:,i],
                    matrix[:,j]
                )
                norm_i = np.linalg.norm(matrix[:,i])
                norm_j = np.linalg.norm(matrix[:,j])

                print('I: ', matrix[:,i])
                print('J: ', matrix[:,j])
                print('Prod: ', inner_product)
                print('Norm i: ', norm_i)
                print('Norm j: ', norm_j)

                if np.abs(inner_product - norm_j * norm_i) < 1E-5:
                    print('Dependent')
                    return True
                else:
                    print('Independent')
                    return False