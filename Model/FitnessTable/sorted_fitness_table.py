import sortedcontainers


class SortedFitnessTable:
    def __init__(self):
        self.table = sortedcontainers.SortedListWithKey(key=lambda val: -val.fitness)

    def add_individual(self, individual):
        self.table.add(individual)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        return self.table[idx]

