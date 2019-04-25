#!/usr/bin/env python
import numpy as np
import gudhi as gd
import pandas as pd
import pickle as pickle
import matplotlib
import typing
import config
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from matplotlib import pyplot as plt
from ripser import Rips, plot_dgms
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def read_data(path: str, columns: int = 1, delimiter: str = ",") -> np.ndarray:
    """
    Reads a certain amount of columns from a .csv-file.
    :param path: Path to the .csv file.
    :param delimiter: Delimiter of the columns within .csv-file. (default: ",")
    :return: Numpy ndarray with columns.
    """
    try:
        if columns == 1:
            data = np.genfromtxt(path, delimiter = delimiter)
        else:
            data = np.genfromtxt(path, delimiter = delimiter)[0:,:columns]

        return data
    except Exception as e:
        raise e


def plot_data(path: str, columns: int = 1, delimiter: str = ",") -> np.ndarray:
    """
    Reads a certain amount of columns from a .csv-file.
    :param path: Path to the .csv file.
    :param delimiter: Delimiter of the columns within .csv-file. (default: ",")
    :return: Numpy ndarray with columns.
    """
    try:
        if columns == 1:
            data = np.genfromtxt(path, delimiter = delimiter)
        else:
            data = np.genfromtxt(path, delimiter = delimiter)[0:,:columns]
            x, y = [], []
            for i in data:
                x.append(i[0])
                y.append(i[1])
            plt.scatter(x,y)
            plt.show()

        return data
    except Exception as e:
        raise e


def sunburst_plot(nodes, total=np.pi * 2, offset=0, level=0, ax=None):
    """
    Plots a sunburst diagram of the data.
    :param nodes: Nodes as a python dict organized hierarchically.
    :param total: Radius of the diagram.
    :param offset: Offset between each of the classes.
    :param level: Level of hierarchy.
    :param ax: Parameter for axes.
    :proc: Plots sunburst diagram, no return value.
    """
    ax = ax or plt.subplot(111, projection='polar')

    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2])
        ax.text(0, 0, label, ha='center', va='center')
        sunburst(subnodes, total=value, level=level + 1, ax=ax)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst(subnodes, total=total, offset=local_offset,
                     level=level + 1, ax=ax)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(values, heights, widths, bottoms, linewidth=1,
                       edgecolor='white', align='edge')
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            ax.text(x, y, label, rotation=rotation, ha='center', va='center')

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_axis_off()

    plt.show()


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
    data = np.genfromtxt(path, delimiter=delimiter)
    diagrams = rips.fit_transform(data, distance_matrix=False)
    rips.plot(diagrams)
    return diagrams


def gudhi_rips_persistence(path: str,
                           columns: int = 1,
                           delimiter: str = ",",
                           max_edge_length: int = 500,
                           max_dimension: int = 3,
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


def make_colormap(seq: float):
    """
    Returns a LinearSegmentedColormap.
    :param seq: A sequence of floats and RGB-tuples. The floats should be increasing and in the interval [0,1].
    :return: LinearSegmentedColormap.
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}

    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def persistence_ring_diagram(path: str,
                             figsize: tuple = (8,8),
                             axes: list = [0.1, 0.1, 0.8, 0.8],
                             sorted: bool = False):
    """
    Plots a persistence ring of some data.
    :param data: N-dimensional numpy array representing any data.
    :param figsize: A tuple with the figure size according to Matplotlib standards.
    :param axes: Position of the polar axes according to Matplotlib.
    :return: Plots a persistence ring diagram (no return value, procedure).
    """

    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes(axes, polar=True)

    # Compute evolution of persistent homology of the dataset as persistence diagram.
    persistence = gudhi_rips_persistence(path, columns = 2, plot = False)
    death, birth = [], []

    for homgroup in persistence:
        if homgroup[1][1] != float("inf"):
            birth.append(homgroup[1][0])
            death.append(homgroup[1][1])
        else:
            pass

    # Sorting the persistence, to yield a suitable representation. (optional)
    if sorted:
        death, birth = zip(* sorted(zip(death, birth)))
        death, birth = (list(t) for t in zip(*sorted(zip(death, birth))))

    N = len(death)
    bottom = birth
    width = 2 * np.pi * np.array(death) / np.sum(death)

    # How many parts does the world of the circle has?
    theta = []
    for i in range(0, len(width)):
        sum = np.sum(width[0:i])
        theta.append(sum)

    # Where should one start with the persistence?
    radii = np.array(death)
    bars = ax.bar(theta, radii, width = width, bottom = bottom, edgecolor = 'black',
                  linewidth=1, align="edge")

    purples = make_colormap(config.HOMOLOGY['colormap']['AvengersEndgame'])
    colorarray = purples(np.linspace(0, 2 * np.pi, N))

    for n,bar in zip(np.arange(N), bars):
        bar.set_facecolor(colorarray[n])

    plt.axis('off')
    plt.show()


########################################################################################################################
""" EXAMPLE OF USAGE
gudhi_rips_persistence("../../data/MOBISIG/USER31/SIGN_FOR_USER31_USER33_10.csv",columns=2, persistence=True)
plot_data("../../data/MOBISIG/USER31/SIGN_FOR_USER31_USER33_10.csv", columns=2)
persistence_ring_diagram("../../data/MOBISIG/USER1/SIGN_FOR_USER1_USER2_2.csv")

Good example files:
../../data/MOBISIG/USER1/SIGN_FOR_USER1_USER2_2.csv
../../data/MOBISIG/USER2/SIGN_FOR_USER2_USER5_14.csv
../../data/MOBISIG/USER16/SIGN_FOR_USER16_USER18_9.csv
../../data/MOBISIG/USER31/SIGN_FOR_USER31_USER33_10.csv
"""
########################################################################################################################