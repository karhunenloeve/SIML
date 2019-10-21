from sklearn.preprocessing import normalize
import numpy as np
import typing
import csv


def retrieve_persistence_distances(
    path: str,
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
            it_0.append(float(i[2]))
        elif "it_1_" in i[1]:
            it_1.append(float(i[2]))
        elif "it_2_" in i[1]:
            it_2.append(float(i[2]))
        elif "it_3_" in i[1]:
            it_3.append(float(i[2]))
        elif "it_4_" in i[1]:
            it_4.append(float(i[2]))

    with open(path + "_retrieved_scores" + ".tex", "w") as myfile:
        wr = csv.writer(
            myfile, quoting=csv.QUOTE_NONE, delimiter="|", quotechar="", escapechar=""
        )
        max_levels, level_counter = len(it_4), 0

        while level_counter < max_levels:
            first_value = it_0[level_counter]
            second_value = it_0[level_counter] + (
                it_1[level_counter] - it_0[level_counter]
            )
            third_value = second_value + (it_2[level_counter] - it_1[level_counter])
            forth_value = third_value + (it_3[level_counter] - it_2[level_counter])
            fifth_value = forth_value + (it_4[level_counter] - it_3[level_counter])

            my_string = (
                "\\addplot+ [mark=none, solid] coordinates{"
                + "(1,0)"
                + "(2,"
                + str(round(first_value, decimals))
                + ")"
                + "(3,"
                + str(round(second_value, decimals))
                + ")"
                + "(4,"
                + str(round(third_value, decimals))
                + ")"
                + "(5,"
                + str(round(forth_value, decimals))
                + ")"
                + "(6,"
                + str(round(fifth_value, decimals))
                + ")};"
            )
            wr.writerow([my_string])
            level_counter += 1

    return print("Retrieval of data finished.")


########################################################################################################################
""" RETRIEVE THE DISTANCES TO TIKZ FILES
Retrieve the experimental results for plots.
These are the Tikz files for the iterational comparison of the respective distances.

retrieve_persistence_distances("results/alpha_wasserstein")
retrieve_persistence_distances("results/rips_wasserstein")
retrieve_persistence_distances("results/witness_wasserstein")
retrieve_persistence_distances("results/alpha_bottleneck")
retrieve_persistence_distances("results/rips_bottleneck")
retrieve_persistence_distances("results/witness_bottleneck")
"""
########################################################################################################################
