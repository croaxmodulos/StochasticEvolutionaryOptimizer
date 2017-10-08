import numbers
import math


class Individual:
    """An "Individual" in the Evolutionary Strategy is fully represented
    by the three  properties: fitness value(scalar), vector of parameters
    and sigma(vector or scalar). Fitness determines the individuals rank,
    compared to other individuals. Parameters is the real-value vector
    to be optimized. Sigma is the real-value vector, responsible for the mutation
    strength(endogenous parameter)."""

    def __init__(self, fitness, parameters, sigma):
        if isinstance(fitness, numbers.Real):
            if math.isfinite(fitness):
                self.fitness = fitness
            else:
                raise ValueError('Fitness value cannot be NaN or Inf')
        elif fitness is None:
            self.fitness = None
        else:
            raise ValueError('Fitness value should be either real or None')

        if len(parameters) == 0:
            raise ValueError('Parameters array cannot be empty')
        else:
            self.params = parameters  # values in range [0.0, 1.0]

        if len(sigma) == 0:
            raise ValueError('Sigma array cannot be empty')
        else:
            self.sigma = sigma

        if len(sigma) != 1:
            if len(sigma) != len(parameters):
                raise ValueError('Sigma size should be either the same as parameters size or one')



