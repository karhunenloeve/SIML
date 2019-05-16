import os
import numpy as np
import random as rand
import matplotlib
import pylab as py
import matplotlib.pyplot as plt
import scipy.interpolate

from matplotlib import cm
from lib import helper as hp
from lib.tda import sim_homology
from scipy.interpolate import Rbf, interp1d, interp2d

def top_nat_neighbors(path: str, column: int=2) -> np.ndarray:
    """
    Nearest neighbor interoplation.
    :param path: Path to the desired CSV-file.
    :param column: Columns to be processed, beginning from the first.
    :return:
    """

    data = hp.read_data(path, column)
    x,y=np.empty(0),np.empty(0)

    for i in data:
        if np.isfinite(i[0]) and np.isfinite(i[1]):
            x = np.append(x,i[0])
            y = np.append(y,i[1])

    xx = np.linspace(np.min(x),np.max(x),len(x))
    f = interp1d(x, y, kind='nearest')
    new_data = []

    for i in range(0,len(xx)):
        new_data.append([xx[i], f(xx[i])])

    return np.array(new_data)


def proc_signatures(dir: str):
    """
    Processes the experiment for the signature dataset.
    Insert the directory to the MOBISID dataset: https://ms.sapientia.ro/~manyi/mobisig.html.
    :param dir: Path to the directory.
    :proc: Directory.
    """
    subdirectories = os.listdir(dir)

    for user_folder in subdirectories:
        if "USER" in user_folder:
            path = os.path.abspath(dir + "/" + user_folder)
            filepaths = os.listdir(path)

            for file in filepaths:
                data = top_nat_neighbors(dir + "/" + user_folder + "/" + file, 2)
                print(data)
                exit(1)


proc_signatures("data/MOBISIG")
#sim_homology.gudhi_rips_persistence("data/MOBISIG/USER16/SIGN_FOR_USER16_USER18_9.csv",2)
#sim_homology.bottleneck_distance("data/MOBISIG/USER16/SIGN_FOR_USER16_USER18_9.csv", "data/MOBISIG/USER31/SIGN_FOR_USER31_USER33_10.csv")


#gudhi_rips_persistence("data/MOBISIG/USER16/SIGN_FOR_USER16_USER18_9.csv",2)