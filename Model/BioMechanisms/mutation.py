import numpy as np
from Model.Individual.individual import Individual


class StandardMutation:
    def __init__(self, tau0, tau1):
        self.tau0 = tau0
        self.tau1 = tau1

    def rnd_std_norm(self, samples):
        return np.random.normal(0, 1, samples)

    def mutate(self, individual):
        """Mutation rule is taken from "Evolution strategies--A comprehensive introduction"
        Beyer et al., 2002, page 29"""

        # mutate sigma
        cmf = np.exp(self.tau0 * self.rnd_std_norm(1))  # common mutation factor for sigma vector
        sigma = cmf * individual.sigma * np.exp(self.tau1 * self.rnd_std_norm(len(individual.sigma)))

        # mutate parameters
        params = individual.params + individual.sigma * self.rnd_std_norm(len(individual.params))

        # in case of overshooting: return parameters inside the range [0.0, 1.0]
        params[params < 0] = 0.0
        params[params > 1] = 1.0

        return Individual(individual.fitness, params, sigma)
