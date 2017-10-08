import unittest
import numpy as np
from ddt import ddt, data, file_data, unpack
from Model.BioMechanisms.selection import select_n_random_unique_parents_from_k


@ddt
class SelectionTest(unittest.TestCase):

    @data([2, 6], [1, 1], [5, 5], [5, 5])
    @unpack
    def test_select_parents(self, n, k):
        indices = select_n_random_unique_parents_from_k(n, k)

        self.assertEqual(n, len(np.unique(indices)))

    def test_select_throws_if_n_is_larger_k(self):
        with self.assertRaises(ValueError):
            (select_n_random_unique_parents_from_k(4, 3))


