import unittest
import numpy as np
from ddt import ddt, data, file_data, unpack
from Model.BioMechanisms.selection import RandomUniqueParentsSelection


@ddt
class SelectionTest(unittest.TestCase):

    def test_select_random_unique_parents_throws(self):
        selector = RandomUniqueParentsSelection(num_new_parents=4)
        with self.assertRaises(ValueError):
            selector.select_random_unique_parents(num_total_parents=3)

    @data([2, 6], [1, 1], [5, 5], [7, 7])
    @unpack
    def test_select_random_unique_parents(self, num_new_parents, num_total_parents):
        selector = RandomUniqueParentsSelection(num_new_parents)
        indices = selector.select_random_unique_parents(num_total_parents)
        self.assertEqual(num_new_parents, len(np.unique(indices)))

