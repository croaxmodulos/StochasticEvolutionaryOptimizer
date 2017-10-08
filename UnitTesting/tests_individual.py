import unittest
from Model.Individual.individual import Individual
import numpy as np
import math


class IndividualTest(unittest.TestCase):
    def test_creation_of_individuals(self):
        individual = Individual(0.5, [-10.0, 20.0, 1000.0], [1.0, 0.5, 0.4])
        self.assertEqual(0.5, individual.fitness)
        self.assertTrue(np.equal(np.array([-10.0, 20.0, 1000.0]), individual.params).all())
        self.assertFalse(np.equal(np.array([-10.0, 20.1, 1000.0]), individual.params).all())
        self.assertTrue(np.equal(np.array([1.0, 0.5, 0.4]),individual.sigma).all())

    def test_create_individual_with_nan_fitness_throws(self):
        with self.assertRaises(ValueError):
            Individual(math.nan, [1.0], [1.0])

    def test_create_individual_with_nan_fitness_throws(self):
        with self.assertRaises(ValueError):
            Individual(math.inf, [1.0], [1.0])

    def test_create_individual_with_complex_fitness_throws(self):
        with self.assertRaises(ValueError):
            Individual(complex(1.0, 1.0), [1.0], [1.0])

    def test_create_individual_with_empty_parameters_throws(self):
        with self.assertRaises(ValueError):
            Individual(-1.0, [], [1.0])

    def test_create_individual_with_empty_sigma_throws(self):
        with self.assertRaises(ValueError):
            Individual(-1.0, [1.0, 2.0], [0.1, 0.2, 0.3])



