import os
import numpy as np
import random as rand
import pylab as py
import matplotlib.pyplot as plt
import scipy.interpolate

from matplotlib import cm
from lib import helper as hp
from lib.tda import sim_homology
from scipy.interpolate import Rbf, interp1d, interp2d
from typing import List, Set, Dict, Tuple, Optional
from multiprocessing import Process


def top_nat_neighbors(
    path: str = "",
    array: np.ndarray = np.empty(1),
    columns: int = 88
) -> np.ndarray:
    """
    Nearest neighbor interpolation.
    Returns the original data with augmented nearest neighbors.
    :param path: Path to the desired CSV-file.
    :param column: Columns to be processed, beginning from the first.
    :return:
    """
    try:
        if len(path) > 0:
            data = hp.read_data(path, columns)
        else:
            data = array
    except ValueError:
        print("Oops! That was no valid number. Try again ...")

    x, y = np.empty(0), np.empty(0)

    for i in data:
        if np.isfinite(i[0]) and np.isfinite(i[1]):
            x = np.append(x, i[0])
            y = np.append(y, i[1])

    xx = np.linspace(np.min(x), np.max(x), len(x))
    f = interp1d(x, y, kind="nearest")
    new_data = []

    for i in range(0, len(xx)):
        new_data.append([xx[i], f(xx[i])])
        new_data.append([x[i], y[i]])

    return np.array(new_data)


def proc_signatures(dir: str, delimiter: str = ",", iterations: int = 5):
    """
    Processes the experiment for the signature dataset.
    Insert the directory to the MOBISID dataset: https://ms.sapientia.ro/~manyi/mobisig.html.
    :param dir: Path to the directory.
    :param delimiter: Delimiter used to save the csv file.
    :proc: Directory.
    """
    subdirectories = os.listdir(dir)

    for user_folder in subdirectories:
        if "USER" in user_folder:
            path = os.path.abspath(dir + "/" + user_folder)
            filepaths = os.listdir(path)

            for file in filepaths:
                temp_data = top_nat_neighbors(
                    path=dir + "/" + user_folder + "/" + file, columns=2
                )

                for j in range(0, iterations):
                    temp_data = top_nat_neighbors(array=temp_data, columns=2)
                    np.savetxt(
                        dir
                        + "/"
                        + "natneighbor"
                        + "/"
                        + user_folder
                        + "/"
                        + "it_"
                        + str(j)
                        + "_"
                        + file,
                        temp_data,
                        delimiter=delimiter,
                    )


def create_distance_file(
    orig_path: str,
    interpol_path: str,
    savefile: bool = True,
    type: ["wasserstein", "bottleneck"] = "wasserstein",
    filtration: ["alpha", "rips", "witness"] = "rips",
    amount_of_files: int = 100
) -> np.ndarray:
    """
    Creates from two directories with corresponding named CSV-files a bottleneck-distance comparison.
    This code relies on the naming of the directories.
    The structure should be: MOBISIG/USERX/file.csv and MOBISIG_natneighbor/USERX/file.csv for a good naming of the .csv rows.
    :param orig_path: Path to the original MOBISIG-files.
    :param interpol_path: Path tot the interpolated MOBISIG-files.
    :param savefile: Whether to save the bottleneck distances into a file or not (npy-format).
    :param amount_of_files: Amount of files to be processed.
    :return: np.ndarray with bottleneck distances.
    """

    def diff(first, second):
        """
        Computes the difference of two list objects.
        :param first: First list.
        :param second: Second list.
        :return: List difference.
        """
        second = set(second)
        return [item for item in first if item not in second]

    original_data, interpolated_data, files_to_ignore = [], [], []

    for dirpath, dirnames, filenames in os.walk(orig_path):
        for filename in filenames:
            files_to_ignore.append(os.path.join(dirpath, filename))
        break
    for dirpath, dirnames, filenames in os.walk(orig_path):
        for filename in filenames:
            original_data.append(os.path.join(dirpath, filename))
    for dirpath, dirnames, filenames in os.walk(interpol_path):
        for filename in filenames:
            interpolated_data.append(os.path.join(dirpath, filename))

    original_data = diff(original_data, files_to_ignore)
    interpolated_data = diff(interpolated_data, files_to_ignore)

    for i in original_data:
        matching = [s for s in interpolated_data if i[20:] in s]
        matching.sort()

        for j in matching:
            distance = sim_homology.persistence_distance(i, j, filtration=filtration, type=type)
            with open("results/" + filtration + "_" + type + ".csv", "a") as fd:
                fd.write(
                    i[20 : len(i) - 4]
                    + ","
                    + j[32 : len(j) - 4]
                    + ","
                    + str(distance)
                    + "\n"
                )

            print(
                "File with name "
                + j
                + " has been compared to "
                + i
                + ". The " + type + "distance is "
                + str(distance)
                + "."
            )


def run_in_parallel(*fns):
    """
    Runs several functions in parallel.
    :param fns: Several functions.
    :return: A nice message.
    """
    proc = []

    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

    return print("Processing finished!")


########################################################################################################################
""" RUN THE DISTANCES
run_in_parallel(
    create_distance_file("data/MOBISIG/", "data/MOBISIG_natneighbor/", filtration="rips", type="wasserstein"),
    create_distance_file("data/MOBISIG/", "data/MOBISIG_natneighbor/", filtration="alpha", type="wasserstein"),
    create_distance_file("data/MOBISIG/", "data/MOBISIG_natneighbor/", filtration="witness", type="wasserstein"),
    create_distance_file("data/MOBISIG/", "data/MOBISIG_natneighbor/", filtration="rips", type="bottleneck"),
    create_distance_file("data/MOBISIG/", "data/MOBISIG_natneighbor/", filtration="alpha", type="bottleneck"),
    create_distance_file("data/MOBISIG/", "data/MOBISIG_natneighbor/", filtration="witness", type="bottleneck")
)
"""
########################################################################################################################
