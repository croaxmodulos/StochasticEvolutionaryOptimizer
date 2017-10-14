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

    def clear(self):
        self.table.clear()

    def remove_last(self):
        if len(self.table) > 0:
            self.table.pop()
