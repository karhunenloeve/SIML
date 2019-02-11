import numpy as np
import gudhi as gd
import pandas as pd
import pickle as pickle
import matplotlib
import typing

from matplotlib import pyplot as plt
from ripser import Rips, plot_dgms
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import genfromtxt
from sys import platform as sys_pf

def read_data(path: str, columns: int, delimiter: str = ",") -> np.ndarray:
    """
    Reads a certain amount of columns from a .csv-file.
    :param path: Path to the .csv file.
    :param delimiter: Delimiter of the columns within .csv-file. (default: ",")
    :return: Numpy ndarray with columns.
    """
    try:
        if rows == 1:
            data = genfromtxt(path, delimiter = delimiter)
        else:
            data = genfromtxt(path, delimiter = delimiter)[0:,:columns]

        return data
    except Exception as e:
        raise e


def makeSparseDM(X, thresh):
    """
    Helper function to make a sparse distance matrix.
    :param X: Dataset to be processed.
    :param thresh: Treshold to be declined.
    :return: Sparse correlation distance matrix.
    """
    N = X.shape[0]
    D = pairwise_distances(X, metric='euclidean')
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= thresh]
    J = J[D <= thresh]
    V = D[D <= thresh]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

def plot_vr_complex(path: str, delimiter: str = ",", thresh: float = 1.0,
                    maxdim: int = 3, coeff = 3, barcode: bool = True) -> np.ndarray:
    """
    Plots the Vietoris Rips complex and returns the data.
    :param path: Path to the desired csv file.
    :param delimiter: Delimiter for the csv file.
    :return: Data for a persistence diagram of a Vietoris Rips complex.
    """
    rips = Rips(maxdim = maxdim, coeff = coeff, do_cocycles = True)
    data = genfromtxt(path, delimiter=delimiter)
    diagrams = rips.fit_transform(data, distance_matrix=False)
    rips.plot(diagrams)
    return diagrams

def gudhi_rips_persistence(path: str,
                           columns: int = 1,
                           delimiter: str = ",",
                           max_edge_length: int = 500,
                           max_dimension: int = 4,
                           barcode: bool = True,
                           persistence: bool = False,
                           plot: bool = True):
    """
    Computes the Vietoris-Rips complex and persistent homology.
    Further it can either plot the barcode, or the persistence diagram.
    :param path: Path to the desired .csv file.
    :param columns: Number of columns to be selected. (default: all)
    :param max_edge_length: Maximal length of an edge within the filtration.
    :param max_dimension: Maximal dimension of a simplex.
    :param delimiter: The delimiter of the .csv-columns.
    :param barcode: Whether plot a barcode diagram or not. (default: True)
    :param persistence: Whether plot a persistence diagram or not. (default: False)
    :param plot: Whether to make a plot or to return values.
    :return: Vietoris-Rips filtration.
    """
    data = read_data(path, columns)
    Rips_complex_sample = gd.RipsComplex(points = data, max_edge_length = max_edge_length)
    Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(max_dimension = max_dimension)
    diag_Rips = Rips_simplex_tree_sample.persistence()

    if barcode and plot:
        gd.plot_persistence_barcode(diag_Rips)
        plt.show()

    if persistence and plot:
        gd.plot_persistence_diagram(diag_Rips)
        plt.show()

    if not plot:
        return diag_Rips

def gudhi_alpha_persistence(path: str,
                            max_alpha_square: float = 0.3,
                            barcode: bool = True,
                            persistence: bool = False,
                            plot: bool = True):
    """
    Computes the Alpha-Complex and persistent homology.
    Further it can either plot the barcode, or the persistence diagram.
    :param max_alpha_square: For each real number n define the concept of a generalized disk of radius 1/n as follow:
                             - If n = 0, it is a closed half-plane;
                             - If n > 0, it is a closed disk of radius 1/n;
                             - If n < 0, it is the closure of the complement of a disk of radius -1/n;
    :param barcode: Whether to create and plot a barcode diagram or not. (default: True)
    :param persistence: Whether plot a persistence diagram or not. (default: False)
    :param plot: Whether to make a plot or to return values.
    :return: Alpha-filtration.
    """
    data = read_data(path, columns)
    Alpha_complex_sample = gd.AlphaComplex(points = data)
    Alpha_simplex_tree_sample = Alpha_complex_sample.create_simplex_tree(max_alpha_square=0.3)
    diag_Alpha = Alpha_simplex_tree_sample.persistence()

    if barcode and plot:
        gd.plot_persistence_barcode(diag_Alpha)
        plt.show()

    if persistence and plot:
        gd.plot_persistence_diagram(diag_Alpha)
        plt.show()

    if not plot:
        return diag_Alpha


gudhi_rips_persistence("../../data/MOBISIG/USER1/SIGN_FOR_USER1_USER2_1.csv", rows = 2)
