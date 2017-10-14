import unittest
import numpy as np
from Model.BioMechanisms.mutation import StandardMutation
from Model.Individual.individual import Individual


class StandardMutationTest(unittest.TestCase):
    def test_after_mutation_parameters_are_within_zero_to_one_range(self):
        individual = Individual(None, 0.5 * np.ones(5), 0.3 * np.ones(5))
        tau0 = 1.0 / np.sqrt(len(individual.params))
        tau1 = 1.0 / np.sqrt(2 * np.sqrt(len(individual.params)))
        m = StandardMutation(tau0, tau1)

        mutated_individual = m.mutate(individual)

        self.assertTrue((mutated_individual.params <= 1.0).all())
        self.assertTrue((mutated_individual.params >= 0.0).all())

