import unittest
import numpy as np
import math
from Model.FitnessFunction.fitness import FitnessEvaluator


class FitnessEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.fitness = FitnessEvaluator(lambda x: np.sum([v ** 2 for v in x]))

    def test_calculate_fitness(self):
        result = self.fitness.compute([2.0, 3.0, 4.0])
        self.assertEqual(29.0, result)

    def test_calculate_fitness_throws(self):
        with self.assertRaises(ValueError):
            self.fitness.compute(np.array([2.0, 3.0, 4.0, math.nan, 5.0]))


if __name__ == '__main__':
    unittest.main()
