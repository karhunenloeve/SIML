
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

def top_nat_neighbors(path: str, column: int) -> np.ndarray:
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

    py.scatter(x, y,color='blue')
    py.scatter(xx, f(xx),color='red')
    plt.show()

    return numpy.empty(1)

sim_homology.gudhi_rips_persistence("data/MOBISIG/USER16/SIGN_FOR_USER16_USER18_9.csv",2)
#gudhi_rips_persistence("data/MOBISIG/USER16/SIGN_FOR_USER16_USER18_9.csv",2)