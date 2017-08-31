import numpy as np


def func(params):
    sigma = 10.0
    result = 1.0

    for item in params:
        result *= (1 - abs(np.sin(2 * item))) * np.exp(- item ** 2 / sigma)

    return result
