import unittest
from Model.FitnessTable.sorted_fitness_table import SortedFitnessTable
from Model.Individual.individual import Individual
import numpy as np


class SortedFitnessTableTest(unittest.TestCase):
    def setUp(self):
        self.table = SortedFitnessTable()

    def test_add_remove_individual_increases_size(self):
        size_initial = len(self.table)

        self.table.add_individual(Individual(5.0, [1.0], [0.1]))

        assert size_initial == 0
        assert len(self.table) == 1

    def test_inserted_elements_are_sorted(self):
        # insert four individuals
        self.table.add_individual(Individual(15.0, [2.0], [0.2]))
        self.table.add_individual(Individual(-15.0, [3.0], [0.3]))
        self.table.add_individual(Individual(-30.0, [4.0], [0.4]))
        self.table.add_individual(Individual(15.0, [5.0], [0.5]))

        self.assertEqual(4, len(self.table))

        for i in range(1, len(self.table)):
            self.assertTrue(self.table[i].fitness <= self.table[i-1].fitness)
