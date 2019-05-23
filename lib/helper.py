#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import typing
import itertools

from sklearn.utils import shuffle
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import FastICA, PCA
from scipy import stats, signal
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.stats.distributions import norm


def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """
    Kernel Density Estimation with Scipy.
    :param x: Data points on the x-axis.
    :param x_grid: Either a grid or the y-axis.
    :param bandwidth: Bandwidth of period.
    :param **kwargs: Type of interpolation.
    :return: Kernel density estimation.
    """
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """
    Univariate Kernel Density Estimation with Statsmodels.
    :param x: Data points on the x-axis.
    :param x_grid: Either a grid or the y-axis.
    :param bandwidth: Bandwidth of period.
    :param **kwargs: Type of interpolation.
    :return: Kernel density estimation.
    """
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """
    Multivariate Kernel Density Estimation with Statsmodels
    :param x: Data points on the x-axis.
    :param x_grid: Either a grid or the y-axis.
    :param bandwidth: Bandwidth of period.
    :param **kwargs: Type of interpolation.
    :return: Kernel density estimation.
    """
    kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x), var_type="c", **kwargs)
    return kde.pdf(x_grid)


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """
    Kernel Density Estimation with Scikit-learn.
    :param x: Data points on the x-axis.
    :param x_grid: Either a grid or the y-axis.
    :param bandwidth: Bandwidth of period.
    :param **kwargs: Type of interpolation.
    :return: Kernel density estimation.
    """
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def findCounts(arr: np.ndarray) -> np.ndarray:
    """
    Replaces the elements of the array by their relative frequencies (only 2D-arrays).
    :param arr: array of the dataset as numpy ndarray.
    :return: numpy ndarray.
    """
    array_shape = arr.shape
    # Check if array is empty, if it is, return 0.
    # This is the trivial case.
    if len(arr) == 0:
        return arr

    # Check if the array is more then two dimensional.
    # If it is, then raise an error.
    if len(array_shape) > 2:
        print("Dimension of ndarray must be exactly two.")
        return False
    else:
        # Init the output list.
        frequencies = np.empty(shape=array_shape, dtype=float)
        # Calculate the statistics for each row.
        for i in range(0, array_shape[1]):
            res = stats.relfreq(arr[:, i], len(arr[:, i]))
            print(res.frequency)
            np.append(frequencies, res.frequency)
    # Returns numpy array with frequencies.
    return frequencies


def read_data(path: str, columns: int = 1, delimiter: str = ",") -> np.ndarray:
    """
    Reads a certain amount of columns from a .csv-file.
    :param path: Path to the .csv file.
    :param delimiter: Delimiter of the columns within .csv-file. (default: ",")
    :return: Numpy ndarray with columns.
    """
    try:
        if columns == 1:
            data = np.genfromtxt(path, delimiter=delimiter)
        else:
            data = np.genfromtxt(path, delimiter=delimiter)[0:, :columns]

        return data
    except Exception as e:
        raise e


def read_data_csv(path: str, sep: str = ",") -> np.ndarray:
    """
    Creates a numpy ndarray from .csv data.
    :param path: path to the csv.
    :return: np.ndarray.
    """
    return np.genfromtxt(path, delimiter=sep)


def standardize(X: np.ndarray) -> np.ndarray:
    """
    Centering the data by subtracting the mean and dividing by standard deviation.
    :param X: An ndarray with the data.
    :return: np.ndarray.
    """
    # Standardize data along the first axis of the tuples.
    X /= X.std(axis=0)
    return X


def compute_ica(S: np.ndarray, n: int) -> dict:
    """
    Computes the FastICA implementation.
    :param S: np.ndarray with data values.
    :param n: Number of independent components to be found.
    :return: {"Signals": S_, "MixingMatrix": A_, "UnmixingMatrix": U_}
    """
    ica = FastICA(n_components=n)
    S_ = ica.fit_transform(S)  # Reconstruct signals / computes ICA.
    A_ = ica.mixing_  # Get estimated mixing matrix.
    U_ = ica.components_  # Get the unmixing matrix.

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
    pca = PCA(n_components=n)
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
        X = read_data_csv(path)  # Reads the path into an np.ndarray.
        s_X = standardize(X)  # Standardizes the data.
        ica_dict = compute_ica(
            s_X, n
        )  # Gets a dict with Data, Signals, MixingMatrix and UnmixingMatrix.
        pca_dict = compute_pca(
            s_X, n
        )  # Gets a dict with Data, Components and SingularValues.

        # Create the plotting object.
        plt.figure()
        models = []  # Here is specified what should be plotted.

        models = [
            ica_dict["Components"],
            ica_dict["MixingMatrix"],
            ica_dict["Signals"],
            pca_dict["Signals"],
            X,
        ]
        names = [
            "Components",
            "Mixing Matrix",
            "ICA recovered signals",
            "PCA recovered signals",
            "Original Signals",
        ]
        colors = [
            "red",
            "steelblue",
            "orange",
            "gray",
            "green",
            "yellow",
            "blue",
            "purple",
        ]

        for ii, (model, name) in enumerate(zip(models, names), 1):
            plt.subplot(5, 1, ii)
            plt.title(name)
            for sig, color in zip(model.T, colors):
                plt.plot(sig, color=color)

        # Adjust the plots to have a nice look.
        plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
        plt.show()
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")


def check_linear_dependence(matrix: np.ndarray) -> bool:
    """
    Functions checks by Cauchy-Schwartz inqeuality whether two matrices are linear dependent or not.
    :param matrix: 2x2 matrix to be processed.
    :return: Boolean.
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i != j:
                inner_product = np.inner(matrix[:, i], matrix[:, j])
                norm_i = np.linalg.norm(matrix[:, i])
                norm_j = np.linalg.norm(matrix[:, j])

                print("I: ", matrix[:, i])
                print("J: ", matrix[:, j])
                print("Prod: ", inner_product)
                print("Norm i: ", norm_i)
                print("Norm j: ", norm_j)

                if np.abs(inner_product - norm_j * norm_i) < 1e-5:
                    print("Dependent")
                    return True
                else:
                    print("Independent")
                    return False


def read_columns_to_dict(path, d=","):
    """
    Write the correlation matrix into a dictionary of nodes.
    :path: fixed path to a csv file.
    :d: delimiter is set to , as default.
    :return: dictionary.
    """
    reader1, reader2 = itertools.tee(csv.reader(path, delimiter=d))
    columns = len(next(reader1))
    rows = len(next(reader2))
    counter_columns, counter_rows = 0, 0

    for i in reader1:
        counter_columns += 1
    for i in reader2:
        counter_rows += 1

    del reader1, reader2
    return counter_columns


def get_power_set(s):
    """
    Computes the powerset lattice of a set.
    :param s: A set.
    :return: A powerset.
    """
    power_set = [set()]

    for element in s:
        new_sets = []
        for subset in power_set:
            new_sets.append(subset | {element})
        power_set.extend(new_sets)

    return power_set


def merge_csv():
    """
    Functions merges two csv files according two columns or rows.
    :return:
    """
    # Todo: Implement the csv merge function
    pass
