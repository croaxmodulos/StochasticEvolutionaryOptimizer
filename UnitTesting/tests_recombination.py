import unittest
import numpy as np
from Model.BioMechanisms.recombination import StandardRecombination
from Model.Individual.individual import Individual


class RecombinationTest(unittest.TestCase):

    def test_recombine_two_parents(self):
        ind1 = Individual(None, [0.11, 0.12, 0.13], [0.011, 0.012, 0.013])
        ind2 = Individual(None, [0.21, 0.22, 0.23], [0.021, 0.022, 0.023])

        newborn = StandardRecombination.recombine([ind1, ind2])

        self.assertTrue(np.equal([0.016, 0.017, 0.018], newborn.sigma).all())

    def test_recombine_threes_parents(self):
        error = 1e-10
        ind1 = Individual(None, [0.11, 0.12, 0.13, 0.14], [0.011, 0.012, 0.013, 0.014])
        ind2 = Individual(None, [0.21, 0.22, 0.23, 0.24], [0.021, 0.022, 0.023, 0.024])
        ind3 = Individual(None, [0.31, 0.32, 0.33, 0.3], [0.031, 0.032, 0.033, 0.034])

        newborn = StandardRecombination.recombine([ind1, ind2, ind3])

        self.assertTrue(np.abs(np.linalg.norm(newborn.sigma - [0.021, 0.022, 0.023, 0.024])) < error)
