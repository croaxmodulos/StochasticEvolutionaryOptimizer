import numpy as np

from Model.BioMechanisms.mutation import StandardMutation
from Model.BioMechanisms.recombination import recombine
from Model.BioMechanisms.selection import select_n_random_unique_parents_from_k
from Model.FitnessTable.sorted_fitness_table import SortedFitnessTable
from Model.Individual.individual import Individual


class OptimizerNGB:
    """Non-Generation-Based(NGB) optimization engine"""

    def __init__(self, params_ngb, mutation, fitness_table=SortedFitnessTable(), ):
        self.params_ngb = params_ngb
        self.mutator = mutation
        self.fitness_table = fitness_table

    def table_random_initialization(self, fitness_object):
        n_params = self.params_ngb.num_params
        initial_sigma = self.params_ngb.initial_sigma

        for i in range(0, self.params_ngb.initial_table_size):
            rnd_individ = Individual(None, np.random.uniform(0.0, 1.0, n_params),
                                     initial_sigma * np.ones(n_params))

            mapped_params = self.map_parameters(rnd_individ.params)  # map the params from range (0.0, 1.0) to search_space

            rnd_individ.fitness = fitness_object.compute(mapped_params)
            self.fitness_table.add_individual(rnd_individ)

    def map_parameters(self, params):
        min_values = self.params_ngb.search_spaces[:, 0]
        max_values = self.params_ngb.search_spaces[:, 1]
        return (max_values - min_values) * params + min_values

    def optimization_start(self, fitness_object):
        self.fitness_table.clear()
        self.table_random_initialization(fitness_object)

        n_parents = self.params_ngb.num_recomb_parents

        for i in range(0, self.params_ngb.max_fitness_calls):
            parents_indices = select_n_random_unique_parents_from_k(n_parents, len(self.fitness_table))
            recombined = recombine([self.fitness_table[x] for x in parents_indices])
            mutated = self.mutator.mutate(recombined)

            mapped_params = self.map_parameters(mutated.params)  # map the params from range (0.0, 1.0) to search_space

            mutated.fitness = fitness_object.compute(mapped_params)
            self.fitness_table.add_individual(mutated)

            if len(self.fitness_table) > self.params_ngb.table_size:
                self.fitness_table.remove_last()



