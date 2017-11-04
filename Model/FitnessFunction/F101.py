import numpy as np


def func(params):
    """F101 function from: http://www.cs.colostate.edu/~genitor/functions.html#f101"""
    sum = 0.0
    dim = len(params)

    for i in range(0, dim - 1):
        x = params[i]
        y = params[i + 1]
        sum += (-x * np.sin(np.sqrt(np.abs(x - (y + 47))))
                - (y + 47) * np.sin(np.sqrt(np.abs(y + 47 + (x / 2)))))

    return sum
