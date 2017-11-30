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

    def clear(self):
        self.records.clear()

    def get_avg_sigma_all_records(self):
        return [r.get_avg_sigma() for r in self.records]

    def get_population_distances_all_records(self):
        return [rec.get_cum_distance_between_individuals() for rec in self.records]


