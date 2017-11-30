import sortedcontainers
import numpy as np


class SortedFitnessTable:
    @classmethod
    def from_table(cls, table):
        obj = cls()
        obj.table = sortedcontainers.SortedListWithKey(table, key=lambda val: -val.fitness)
        return obj

    def __init__(self):
        self.table = sortedcontainers.SortedListWithKey(key=lambda val: -val.fitness)

    def add_individual(self, individual):
        self.table.add(individual)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        return self.table[idx]

    def clear(self):
        self.table.clear()

    def remove_last(self):
        if len(self.table) > 0:
            self.table.pop()

    def get_avg_sigma(self):
        return np.sum([np.mean(e.sigma) for e in self.table]) / len(self.table)

    def get_cum_distance_between_individuals(self):
        d = 0.0
        for i in range(0, len(self.table)):
            for j in range(0, len(self.table)):
                d += np.linalg.norm(self.table[i].params - self.table[j].params)
        return d
