import numpy as np
from Model.FitnessTable.sorted_fitness_table import SortedFitnessTable
from Model.Individual.individual import Individual


class OptimizerNGB:
    """Non-Generation-Based(NGB) optimization engine"""

    def __init__(self, params_ngb, selection, recombination,
                 mutation, statistics_recorder):
        self.params_ngb = params_ngb
        self.select_from = selection
        self.recombine = recombination
        self.mutate = mutation
        self.fitness_table = SortedFitnessTable()
        self.statistics_recorder = statistics_recorder

    def table_random_initialization(self, fitness_object):
        n_params = self.params_ngb.num_params
        initial_sigma = self.params_ngb.initial_sigma
        sigma_dim = n_params if self.params_ngb.is_sigma_array else 1

        for i in range(0, self.params_ngb.initial_table_size):
            rnd_individ = Individual(None, np.random.uniform(0.0, 1.0, n_params),
                                     initial_sigma * np.ones(sigma_dim))

            mapped_params = self.map_parameters(rnd_individ.params)  # map params from (0.0, 1.0) -> search_space

            rnd_individ.fitness = fitness_object.compute(mapped_params)
            self.fitness_table.add_individual(rnd_individ)

    def map_parameters(self, params):
        min_values = self.params_ngb.search_spaces[:, 0]
        max_values = self.params_ngb.search_spaces[:, 1]
        return (max_values - min_values) * params + min_values

    def optimization_start(self, fitness_object):
        # self.print_fitness_table()
        self.fitness_table.clear()
        self.table_random_initialization(fitness_object)

        if self.statistics_recorder is not None:
            self.statistics_recorder.record(iteration=0, fitness_table=self.fitness_table)

        # self.print_fitness_table()

        for i in range(1, self.params_ngb.max_fitness_calls + 1):
            parents_indices = self.select_from(len(self.fitness_table))
            recombined = self.recombine([self.fitness_table[x] for x in parents_indices])
            mutated = self.mutate(recombined)

            mapped_params = self.map_parameters(mutated.params)  # map the params from range (0.0, 1.0) to search_space

            mutated.fitness = fitness_object.compute(mapped_params)
            self.fitness_table.add_individual(mutated)

            # self.print_fitness_table()

            if len(self.fitness_table) > self.params_ngb.table_size:
                self.fitness_table.remove_last()

            if self.statistics_recorder is not None:
                self.statistics_recorder.record(i, self.fitness_table)

        return self.fitness_table, self.statistics_recorder

    def print_fitness_table(self):
        for i in range(0, len(self.fitness_table)):
            print("{0:.3e}, {1}, {2}".format(self.fitness_table[i].fitness,
                                             self.fitness_table[i].params,
                                             self.fitness_table[i].sigma))

        print("-------------------------------------------------------------")
