import math


class FitnessEvaluator:
    def __init__(self, fun):
        """Construct class with a function object (functor), functor is used in 'compute' method"""

        self.functor = fun

    def compute(self, params):
        """Given an input array of parameters, a scalar value is returned (fitness value)

        Args:
            params: 1d numpy array of numbers (e.g. int, float)

        Returns:
            float: fitness value

        Raises:
            ValueError: if params contain NaN elements

        Examples:
            >>> import numpy as np
            >>> fitness = FitnessEvaluator(lambda x: np.sum(x))
            >>> print(fitness.compute([1.0, 2.0, 3.0]))
            6.0

        """

        for item in params:
            if math.isnan(item):
                raise ValueError('Input parameters cannot be NaN')

        return self.functor(params)
