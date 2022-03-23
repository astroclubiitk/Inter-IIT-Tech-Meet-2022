import numpy as np

K = 0.5

"""Return the slope calculated between two data points in 2-Dimensional space"""


def get_slope(x1, x2, y1, y2):
    return (y1 - y2) / (x1 - x2)


"""Return the Euclidean distance between two points"""


def euclidean(x1, x2, y1, y2):
    _y = (y1 - y2) * (y1 - y2)
    _x = (x1 - x2) * (x1 - x2)
    return np.sqrt(_x + _y)


"""Helper functions to fit on the detected flares"""


def exp_fit_func(x, ln_a, b):
    t = x**K
    return ln_a - b * t


def exp_func(x, a, b):
    t = -1 * b * (x**K)
    return a * np.exp(t)


def inverse_exp_func(y, a, b):
    t1 = np.log(y) - np.log(a)
    t2 = -1 * t1 / b
    return int(t2 ** (1.0 / K))
