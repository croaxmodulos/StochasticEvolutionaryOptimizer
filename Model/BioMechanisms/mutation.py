import numpy as np
from Model.Individual.individual import Individual


class StandardMutation:
    """Mutation rule is taken from "Evolution strategies--A comprehensive introduction"
    Beyer et al., 2002, page 29"""

    def __init__(self, num_params):
        self.tau0 = 1.0 / np.sqrt(2 * num_params)
        self.tau1 = 1.0 / np.sqrt(2 * np.sqrt(num_params))

    @staticmethod
    def rnd_std_norm(samples):
        return np.random.normal(0, 1, samples)

    def mutate(self, individual):
        # mutate sigma array
        cmf = np.exp(self.tau0 * self.rnd_std_norm(1))  # common mutation factor for sigma vector
        sigma = cmf * individual.sigma * np.exp(self.tau1 * self.rnd_std_norm(len(individual.sigma)))

        # mutate parameters, use newly mutated sigma (important)
        params = individual.params + sigma * self.rnd_std_norm(len(individual.params))

        # in case of overshooting: return parameters inside the range [0.0, 1.0]
        params[params < 0.0] = 0.0
        params[params > 1.0] = 1.0

        return Individual(individual.fitness, params, sigma)
