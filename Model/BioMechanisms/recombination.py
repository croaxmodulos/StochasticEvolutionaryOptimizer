import random
import numpy as np

from Model.Individual.individual import Individual


def recombine(individuals):
    """As a result of the parents recombination a new child is produced.
    Recombination schemes of the parameters and sigmas are generally different.
    Each parameter of the newborn(recombined) individual
    represents a copy from one of the parents parameters. Parent for each parameter
    is chosen randomly. Sigma of the recombined individual is a simple average"""

    parents = len(individuals)
    params = np.zeros(len(individuals[0].params))
    sigma = np.zeros(len(individuals[0].sigma))

    for i in range(0, len(params)):
        idx = random.randint(0, parents - 1)
        params[i] = individuals[idx].params[i]

    for i in range(0, len(sigma)):
        for j in range(0, parents):
            sigma[i] += individuals[j].sigma[i]

    sigma /= parents

    return Individual(None, params, sigma)
