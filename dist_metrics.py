import numpy as np


def minkowski_distance(x, y, weights, m=2, norm=False):
    """
    Minkowski distance between two points.
    :param weights: weights values for each of the instances
    :param x: numpy array of datapoint x
    :param y: numpy array of datapoint y
    :param m: minkowski value
    :return: the Minkowski distance between x and y
    """

    if norm:
        dist = np.sum(np.abs(x) ** m) ** (1 / m)
    else:
        dist = np.sum(np.multiply(np.abs(y-x), weights) ** m, axis=1) ** (1 / m)

    return dist


def cos_distance(x, y, weights):
    """
    Cosine distance between two points.
    :param weights: weights values for each of the instances
    :param x: numpy array of datapoint x
    :param y: numpy array of datapoint y
    :return: the cosine distance between x and y
    """
    cosine = np.sum(np.multiply((x-y), weights), axis=1) / (
            np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2, axis=1)))
    return cosine


def HasD(x, y, weights):
    """
    Hassanat distance between two points
    :param weights: weights values for each of the instances
    :param x: numpy array of datapoint x
    :param y: numpy array of datapoint y
    :return: the Hassanat distance between x and y
    """
    res_array = []

    for array in y:
        total = 0
        for i in range(len(array)):
            xi = x[i]
            yi = array[i]
            w = weights[i]

            min_value = min(xi, yi)
            max_value = max(xi, yi)

            if min_value >= 0:
                total += (1 - ((1 + min_value) / (1 + max_value)))*w
            else:
                total += (1 - ((1 + min_value + abs(min_value)) / (1 + max_value + abs(min_value))))*w
        res_array.append(total)
    return np.array(res_array, dtype='float64')
