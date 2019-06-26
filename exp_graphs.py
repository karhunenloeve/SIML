from numpy import genfromtxt
import numpy as np
import typing
import csv


def retrieve_bttlnck_distances(path: str,
                               delimiter: str = ",",
                               decimals: int = 2):
    """
    Customized retrieval function for the bottleneck distances.
    Retrieves elements from a 3-tuple, named by the iteration step it_i.
    Works for CSV files only.
    :param path: Path of the respective file.
    :param delimiter: CSV delimiter, by default set to ','.
    :return: A nice message to communicate that we are done.
    """
    with open(path + ".csv", "r") as f:
        reader = csv.reader(f)
        my_list = list(reader)

    it_0, it_1, it_2, it_3, it_4 = [], [], [], [], []
    for i in my_list:
        if "it_0_" in i[1]:
            it_0.append(i[2])
        elif "it_1_" in i[1]:
            it_1.append(i[2])
        elif "it_2_" in i[1]:
            it_2.append(i[2])
        elif "it_3_" in i[1]:
            it_3.append(i[2])
        elif "it_4_" in i[1]:
            it_4.append(i[2])

    with open(path + "_retrieved_scores" + ".csv", "w") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        max_levels, level_counter = len(it_4), 0

        while level_counter < max_levels:
            wr.writerow(
                (
                    round(float(it_0[level_counter]),decimals),
                    round(float(it_1[level_counter]),decimals),
                    round(float(it_2[level_counter]),decimals),
                    round(float(it_3[level_counter]),decimals),
                    round(float(it_4[level_counter]),decimals),
                )
            )
            level_counter += 1

    return print("Retrieval of data finished.")


retrieve_bttlnck_distances("results/alpha_bottleneck")
retrieve_bttlnck_distances("results/rips_bottleneck")
retrieve_bttlnck_distances("results/witness_bottleneck")
