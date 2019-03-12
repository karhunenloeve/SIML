import lib.antcolony
import csv
import math
import scipy
import numpy as np
from typing import AnyStr, Callable
from lib import metrics as mtr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import pdist, squareform

def create_adjacency_matrix(path: list, distfkt: Callable) -> dict:
    """
    Creates an adjacency matrix for all columns of a dataset.
    :param path: Path to the desired csv file.
    :param distfkt: Distance function.
    :return: Adjacency matrix as dictionary of the form {"1_to_8": distcorr()}.
    """

    with open(path, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    columns, rows = len(data_list[0]), len(data_list)
    adj_matrix = {}

    for i in range(0, columns):
        for j in range(0, columns):
            adj_matrix[str(i) + "_to_" + str(j)] = distfkt(data_list[i], data_list[j])

    return adj_matrix

print(create_adjacency_matrix("data/abalone.csv", mtr.distcorr))
# Todo: finish experiments of functional dependencies
# Todo: implementation and visualization of graphtools for persistent homology
# Todo: comparison of persistent homology and topological clustering