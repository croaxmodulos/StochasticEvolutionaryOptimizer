import random
import numpy as np
from Model.Individual.individual import Individual


class StandardRecombination:
    @staticmethod
    def recombine(individuals):
        """As a result of the parents recombination a new child is produced.
        Each parameter of the newborn(recombined) individual represents a copy
        of one of the parent's. Parent for each parameter is chosen randomly.
        Sigma of the recombined individual is a simple average of all parents"""

        parents = len(individuals)
        params = np.zeros(len(individuals[0].params))
        sigma = np.zeros(len(individuals[0].sigma))

        # recombine parameters array
        for i in range(0, len(params)):
            idx = random.randint(0, parents - 1)
            params[i] = individuals[idx].params[i]

        # recombine sigma array
        for i in range(0, len(sigma)):
            for j in range(0, parents):
                sigma[i] += individuals[j].sigma[i]
        sigma /= parents

        return Individual(None, params, sigma)
