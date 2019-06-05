#!/usr/bin/env python
import numpy as np
import gudhi as gd
import matplotlib.colors as mcolors

from handler.timeout import timeout
from matplotlib import pyplot as plt
from ripser import Rips


c = mcolors.ColorConverter().to_rgb

HOMOLOGY = {
    "colormap": {
        "IronRed": {
            c("#f6e6e6"),
            c("#e5b4b4"),
            c("#d48282"),
            c("#c35050"),
            c("#b21e1e"),
            c("#aa0505"),
        },
        "IronBlue":{
            c("#4b8ac7"),
            c("#5299d0"),
            c("#5299d0"),
            c("#5aaad7"),
            c("#61bbe4"),
            c("#64bee6"),
        },
        "IronYellow":{
            c("#FEF9E5"),
            c("#FDE99A"),
            c("#FCD94E"),
            c("#FBD435"),
            c("#fbca03"),
        },
        "AvengersEndgame": [
            c("#0B0930"),
            c("#1A1A64"),
            c("#2C2A89"),
            c("#453AA4"),
            c("#5C49C6"),
            c("#7B6FDE"),
        ],
        "IronMan": [
            c("#AA0505"),
            c("#6A0C0B"),
            c("#B97D10"),
            c("#FBCA03"),
            c("#67C7EB"),
        ],
    }
}


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


def plot_data(path: str, columns: int = 1, delimiter: str = ",") -> np.ndarray:
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
            x, y = [], []
            for i in data:
                x.append(i[0])
                y.append(i[1])
            plt.scatter(x, y)
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
    ax = ax or plt.subplot(111, projection="polar")

    if level == 0 and len(nodes) == 1:
        label, value, subnodes = nodes[0]
        ax.bar([0], [0.5], [np.pi * 2])
        ax.text(0, 0, label, ha="center", va="center")
        sunburst(subnodes, total=value, level=level + 1, ax=ax)
    elif nodes:
        d = np.pi * 2 / total
        labels = []
        widths = []
        local_offset = offset
        for label, value, subnodes in nodes:
            labels.append(label)
            widths.append(value * d)
            sunburst(subnodes, total=total, offset=local_offset, level=level + 1, ax=ax)
            local_offset += value
        values = np.cumsum([offset * d] + widths[:-1])
        heights = [1] * len(nodes)
        bottoms = np.zeros(len(nodes)) + level - 0.5
        rects = ax.bar(
            values,
            heights,
            widths,
            bottoms,
            linewidth=1,
            edgecolor="white",
            align="edge",
        )
        for rect, label in zip(rects, labels):
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_y() + rect.get_height() / 2
            rotation = (90 + (360 - np.degrees(x) % 180)) % 360
            ax.text(x, y, label, rotation=rotation, ha="center", va="center")

    if level == 0:
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
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
    D = pairwise_distances(X, metric="euclidean")
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= thresh]
    J = J[D <= thresh]
    V = D[D <= thresh]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()


def plot_vr_complex(
    path: str,
    delimiter: str = ",",
    thresh: float = 1.0,
    maxdim: int = 3,
    coeff=3,
    barcode: bool = True,
) -> np.ndarray:
    """
    Plots the Vietoris Rips complex and returns the data.
    :param path: Path to the desired csv file.
    :param delimiter: Delimiter for the csv file.
    :return: Data for a persistence diagram of a Vietoris Rips complex.
    """
    rips = Rips(maxdim=maxdim, coeff=coeff, do_cocycles=True)
    data = np.genfromtxt(path, delimiter=delimiter)
    diagrams = rips.fit_transform(data, distance_matrix=False)
    rips.plot(diagrams)
    return diagrams


def gudhi_rips_persistence(
    path: str,
    columns: int = 1,
    delimiter: str = ",",
    max_edge_length: int = 500,
    max_dimension: int = 3,
    barcode: bool = True,
    persistence: bool = False,
    plot: bool = True,
):
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
    Rips_complex_sample = gd.RipsComplex(points=data, max_edge_length=max_edge_length)
    Rips_simplex_tree_sample = Rips_complex_sample.create_simplex_tree(
        max_dimension=max_dimension
    )
    diag_Rips = Rips_simplex_tree_sample.persistence()

    if barcode and plot:
        gd.plot_persistence_barcode(diag_Rips)
        plt.show()

    if persistence and plot:
        gd.plot_persistence_diagram(diag_Rips)
        plt.show()

    if not plot:
        return diag_Rips


def gudhi_alpha_persistence(
    path: str,
    columns: int = 1,
    max_alpha_square: float = 0.3,
    barcode: bool = True,
    persistence: bool = False,
    plot: bool = True,
):
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
    Alpha_complex_sample = gd.AlphaComplex(points=data)
    Alpha_simplex_tree_sample = Alpha_complex_sample.create_simplex_tree(
        max_alpha_square=0.3
    )
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
    cdict = {"red": [], "green": [], "blue": []}

    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap("CustomMap", cdict)


def persistence_ring_diagram_tikz(
    path: str, save: bool = True, filename: str = "tikz.tex", delta: float = 0.2
):
    """
    Creates a ring diagram as Tikz image.
    :param path: Path of the csv file.
    :param filename: Filename.
    :return: Diagram as tikz.

    Tikz-code for drawsector:
    \newcommand{\drawsector}[6][]{
    \draw[#1] (#4:{#2-.5*#3}) arc [start angle = #4, delta angle=-#5, radius={#2-.5*#3}]--++({#4-#5}:#3) arc [start angle = {#4- #5}, delta angle=#5, radius={#2+.5*#3}] --cycle;
    \draw[decorate,decoration={raise=-3pt, text along path, text=#6, text align={align=center}}] (#4:#2) arc(#4:(#4-#5):#2);}
    """
    persistence = gudhi_rips_persistence(path, columns=2, plot=False)
    death, birth, hom = [], [], []

    for homgroup in persistence:
        if homgroup[1][1] != float("inf"):
            birth.append(homgroup[1][0])
            death.append(homgroup[1][1])
            hom.append(homgroup[0])
        else:
            pass

    birth, death, hom = (
        np.array(list(reversed(birth))),
        np.array(list(reversed(death))),
        np.array(list(reversed(hom))),
    )

    diff = death - birth
    start_angle = birth / np.sum(birth) * 360
    delta_angle = death / np.sum(death) * 360

    width = diff / np.sum(diff) * 3
    radius = []
    for i in range(0,len(width)):
        radius.append(np.sum(width[0:i]))

    diagram = "\\begin{tikzpicture} \n"
    colors = ["lightcandy", "lightblue", "lightgold"]

    for i in range(0, len(death)):
        if hom[i] == 0:
            color = colors[0]
        elif hom[i] == 1:
            color = colors[1]
        else:
            color = colors[2]
        intensity = 100 - (round((100 - 10) * i / len(death) + 10))
        diagram = (
            diagram
            + "\t \\drawsector[draw=black, fill=white!"
            + str(intensity)
            + "!"
            + color
            + "]{"
            + str(round(radius[i], 1))
            + "}{"
            + str(round(width[i], 1))
            + "}{"
            + str(round(start_angle[i], 1))
            + "}{"
            + str(round(delta_angle[i], 1))
            + "}{\\empty} \n"
        )

    diagram = diagram + "\\end{tikzpicture}"

    if save == True:
        with open(filename, "w") as text_file:
            text_file.write(diagram)

    return diagram


def persistence_ring_diagram(
    path: str,
    figsize: tuple = (8, 8),
    axes: list = [0.1, 0.1, 0.8, 0.8],
    sorted: bool = False,
    hom: int = 1,
    map: str = "IronRed"
):
    """
    Plots a persistence ring of some data.
    :param data: N-dimensional numpy array representing any data.
    :param figsize: A tuple with the figure size according to Matplotlib standards.
    :param axes: Position of the polar axes according to Matplotlib.
    :return: Plots a persistence ring diagram (no return value, procedure).
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(axes, polar=True)

    # Compute evolution of persistent homology of the dataset as persistence diagram.
    persistence = gudhi_rips_persistence(path, columns=2, plot=False)
    death, birth, homgrp = [], [], []

    for homgroup in persistence:
        if homgroup[1][1] != float("inf") and homgroup[0] == hom:
            birth.append(homgroup[1][0])
            death.append(homgroup[1][1])
            homgrp.append(homgroup[0])
        else:
            pass

    # Sorting the persistence, to yield a suitable representation. (optional)
    if sorted:
        death, birth = zip(*sorted(zip(death, birth)))
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
    bars = ax.bar(
        theta,
        radii,
        width=width,
        bottom=bottom,
        edgecolor="black",
        linewidth=1,
        align="edge",
    )

    purples = make_colormap(HOMOLOGY["colormap"][map])
    colorarray = purples(np.linspace(0, 2 * np.pi, N))

    for n, bar in zip(np.arange(N), bars):
        bar.set_facecolor(colorarray[n])

    plt.axis("off")
    plt.show()


@timeout(seconds=10 ** 3)
def bottleneck_distance(
    path1: str,
    path2: str,
    delimiter: str = ",",
    columns: int = 2,
    max_edge_length: int = 200.0,
    max_dimension: int = 1,
    landmark_percentage=1,
    filtration: ["alpha", "rips", "witness"] = "rips",
) -> float:
    """
    :param path1: Path of the first csv file.
    :param path2: Path of the second csv file.
    :param max_edge_length: Maximal length of an edge within the filtration.
    :param max_dimension: Maximal dimension of a simplex.
    :param columns: Columns to be spanned inside the filtration.
    :param filtration: Which filtration has to be choosen.
    :return: The bottleneck distance between two diagrams.
    """
    data1 = read_data(path1, columns)
    data2 = read_data(path2, columns)

    nans1 = np.argwhere(np.isnan(data1))
    nans2 = np.argwhere(np.isnan(data2))

    for i in nans1:
        data1 = np.delete(data1, nans1)
    for j in nans2:
        data2 = np.delete(data2, nans2)

    diag_1, diag_2 = [], []
    data1 = data1.reshape((int(data1.size / 2), 2))
    data2 = data2.reshape((int(data2.size / 2), 2))

    if filtration == "alpha":
        # First sample processed.
        complex_sample1 = gd.AlphaComplex(points=data1)
        complex_tree_sample1 = complex_sample1.create_simplex_tree()
        diag1 = complex_tree_sample1.persistence()
        # Second sample processed.
        complex_sample2 = gd.AlphaComplex(points=data2)
        complex_tree_sample2 = complex_sample2.create_simplex_tree()
        diag2 = complex_tree_sample2.persistence()
    elif filtration == "rips":
        # First sample processed.
        complex_sample1 = gd.RipsComplex(points=data1, max_edge_length=max_edge_length)
        complex_tree_sample1 = complex_sample1.create_simplex_tree(
            max_dimension=max_dimension
        )
        diag1 = complex_tree_sample1.persistence()
        # Second sample processed.
        complex_sample2 = gd.RipsComplex(points=data2, max_edge_length=max_edge_length)
        complex_tree_sample2 = complex_sample2.create_simplex_tree(
            max_dimension=max_dimension
        )
        diag2 = complex_tree_sample2.persistence()
    elif filtration == "witness":
        # First sample processed.
        landmarks = gd.pick_n_random_points(
            points=data1, nb_points=round(data1.size / 100 * landmark_percentage)
        )
        witness_complex = gd.EuclideanStrongWitnessComplex(
            witnesses=data1, landmarks=landmarks
        )
        complex_tree_sample1 = witness_complex.create_simplex_tree(
            max_alpha_square=10 ** 3, limit_dimension=max_dimension
        )
        diag1 = complex_tree_sample1.persistence()
        # Second sample processed.
        landmarks = gd.pick_n_random_points(
            points=data2, nb_points=round(data1.size / 100 * landmark_percentage)
        )
        witness_complex = gd.EuclideanStrongWitnessComplex(
            witnesses=data2, landmarks=landmarks
        )
        complex_tree_sample2 = witness_complex.create_simplex_tree(
            max_alpha_square=10 ** 3, limit_dimension=max_dimension
        )
        diag2 = complex_tree_sample1.persistence()
    else:
        print("Wrong filtration specified.")
        exit(1)

    # Rebuilding objects for bottleneck distance calculation.
    for i in range(1, max(len(diag1), len(diag2))):
        if i < len(diag1):
            element1 = [diag1[i][1][0], diag1[i][1][1]]
            diag_1.append(element1)
        if i < len(diag2):
            element2 = [diag2[i][1][0], diag2[i][1][1]]
            diag_2.append(element2)

    distance = gd.bottleneck_distance(diag_1, diag_2)
    print("The diagrams distance is: " + str(distance) + " bttlnck.")
    return distance


########################################################################################################################
""" EXAMPLE OF USAGE
gudhi_rips_persistence("../../data/MOBISIG/USER31/SIGN_FOR_USER31_USER33_10.csv",columns=2, persistence=True)
plot_data("../../data/MOBISIG/USER31/SIGN_FOR_USER31_USER33_10.csv", columns=2)
persistence_ring_diagram("../../data/MOBISIG/USER1/SIGN_FOR_USER1_USER2_2.csv")

Good example files:
../../data/MOBISIG/USER01/SIGN_FOR_USER1_USER2_2.csv
../../data/MOBISIG/USER02/SIGN_FOR_USER2_USER5_14.csv
../../data/MOBISIG/USER16/SIGN_FOR_USER16_USER18_9.csv
../../data/MOBISIG/USER31/SIGN_FOR_USER31_USER33_10.csv
"""
########################################################################################################################