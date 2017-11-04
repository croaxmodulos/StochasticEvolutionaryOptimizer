from sortedcontainers import SortedListWithKey
from Model.FitnessTable.sorted_fitness_table import SortedFitnessTable


class StatisticsRecorder:
    def __init__(self, record_after_iterations):
        self.record_after_iterations = record_after_iterations
        self.records = []  # list of "pools"(fitness tables)

    def record(self, iteration, fitness_table):
        if (iteration % self.record_after_iterations) == 0:
            t = SortedFitnessTable.from_table(fitness_table)
            self.records.append(t)

