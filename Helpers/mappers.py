import numpy as np


def map_linearly_from_to(values, from_range, to_ranges):
    """Given the array of values [x_1, ..., x_i, ..., x_n],
    where x_i is defined in the range [from_range[0], from_range[1]],
    map all x_i linearly to another set of ranges [to_ranges[i][0], to_ranges[i][1]]

    Transformation is linear:
    y = a*x + b

    Thus, a system of two equations for a and b should be solved:
    to_min = a * from_min + b
    to_max = a * from_max + b
    """

    mapped = np.zeros(len(values))
    for idx, x in enumerate(values):
        a = (to_ranges[idx][0] - to_ranges[idx][1]) / (from_range[0] - from_range[1])
        b = to_ranges[idx][0] - a * from_range[0]
        mapped[idx] = a * x + b

    return mapped
