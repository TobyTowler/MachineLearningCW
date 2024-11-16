# Description
# Created on 16/10/24
# ---------------------


"""

GAINS

21.203398393975398
21.203398393975398
21.203398393975398

GINI

-39.5
-71.5
-71.5

CHI

10.213333333333333
2.8888888888888884
2.8888888888888884

"""

# ---------------------------
# Attribuite: Headache
# Contingency table:
# [[3, 0], [2, 3]]
# --------------------------
# Infomation gain
#

import math
from warnings import warn
import numpy as np


def get_gain(contingency_table):
    """
    Function to calculate the infomation gain of an attribute

    Paramaters:
    contingency_table : 2x2 list
        the contingency_table for the attribute to be assessed

    Returns
    -------
    string
        error message
    float
        infomation gain
    """

    if contingency_table[0] is None or contingency_table[1] is None:
        return "invalid input"

    rootTrue = contingency_table[0][0] + contingency_table[0][1]

    total = sum(contingency_table[0]) + sum(contingency_table[1])

    def calcGain(root, true):
        false = total - true
        return root - (true * math.log(true, 2) + false * math.log(false, 2))

    root = calcGain(0, rootTrue)
    gainT = calcGain(root, sum(contingency_table[0]) / total)
    gainF = calcGain(root, sum(contingency_table[1]) / total)

    gain = root - (
        (sum(contingency_table[0]) / total * gainT)
        + (sum(contingency_table[1]) / total * gainF)
    )
    return gain


def get_gini(contingency_table):
    """
    Function to calculate the gini impurity of an attribute

    Paramaters:
    contingency_table : 2x2 list
        the contingency_table for the attribute to be assessed

    Returns
    -------
    string
        error message
    float
        gini impurity
    """

    if contingency_table[0] is None or contingency_table[1] is None:
        return "invalid input"

    rootTrue = contingency_table[0][0] + contingency_table[0][1]

    total = sum(contingency_table[0]) + sum(contingency_table[1])

    def calcGini(true):
        false = total - true
        return 1 - (true**2) + (false**2)

    root = calcGini(rootTrue)
    giniT = calcGini(sum(contingency_table[0]) / total)
    giniF = calcGini(sum(contingency_table[1]) / total)

    gini = root - (
        (sum(contingency_table[0]) / total * giniT)
        + (sum(contingency_table[1]) / total * giniF)
    )
    return gini


def get_chi(contingency_table):
    """
    Function to calculate the chi-square statistic of an attribute

    Paramaters:
    contingency_table : 2x2 list
        the contingency_table for the attribute to be assessed

    Returns
    -------
    string
        error message
    float
        chi-square statistic
    """

    if contingency_table[0] is None or contingency_table[1] is None:
        return "invalid input"

    rootTrue = contingency_table[0][0] + contingency_table[0][1]

    table = [[-1, -1], [-1, -1]]

    for x in range(0, 2):
        for y in range(0, 2):
            rowTotal = sum(contingency_table[y])
            colTotal = contingency_table[x][0] + contingency_table[x][1]
            try:
                colProb = colTotal / rootTrue
            except:
                return "0 error in table"
            table[x][y] = rowTotal * colProb

    x2 = 0
    for i in range(0, len(contingency_table[0])):
        for j in range(0, len(contingency_table[0])):
            x2 += ((contingency_table[i][j] - table[i][j]) ** 2) / table[i][j]

    return x2


def main():
    HeadacheDiagnosis = [[3, 0], [2, 3]]
    SpotsDiagnosis = [[4, 1], [1, 2]]
    StiffneckDiagnosis = [[4, 1], [1, 2]]

    print("\nGAINS\n")
    print(get_gain(HeadacheDiagnosis))
    print(get_gain(SpotsDiagnosis))
    print(get_gain(StiffneckDiagnosis))

    print("\nGINI\n")
    print(get_gini(HeadacheDiagnosis))
    print(get_gini(SpotsDiagnosis))
    print(get_gini(StiffneckDiagnosis))

    print("\nCHI\n")
    print(get_chi(HeadacheDiagnosis))
    print(get_chi(SpotsDiagnosis))
    print(get_chi(StiffneckDiagnosis))


main()
