import unittest
from events import Events

from Helpers.statistics_recorder import StatisticsRecorder
from Model.FitnessTable.sorted_fitness_table import SortedFitnessTable


class StatisticsRecorderTest(unittest.TestCase):
    @staticmethod
    def function_which_throws():
        raise ValueError

    def test_handler_throws_exception(self):
        events = Events()
        events.on_change += self.function_which_throws
        with self.assertRaises(ValueError):
            events.on_change()

    def test_recorder_records_four_iterations(self):
        table = SortedFitnessTable()
        r = StatisticsRecorder(record_after_iterations=3)
        events = Events()
        events.on_change += r.record
        it_total = 10  # 0, 3, 6, 9 should be recorded
        expected_records = 4

        for it in range(it_total):
            events.on_change(it, table)

        self.assertEquals(expected_records, len(r.records))