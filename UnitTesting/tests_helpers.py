import unittest
from ddt import ddt, data, file_data, unpack
from Helpers.mappers import map_linearly_from_to
import numpy as np


@ddt
class MapperTest(unittest.TestCase):

    @data([[-1.2, 1.5, 20.0], [0.0, 0.5, 1.0], [0.0, 1.0], [[-1.2, 2.8], [-2.0, 5.0], [10.0, 20.0]]])
    @unpack
    def test_linear_mapping(self, expectations, values, from_range, to_ranges):
        mapped = map_linearly_from_to(values, from_range, to_ranges)
        self.assertTrue(np.equal(expectations, mapped).all())

