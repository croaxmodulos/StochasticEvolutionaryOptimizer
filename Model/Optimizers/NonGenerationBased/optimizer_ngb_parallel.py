import multiprocessing
import threading
import numpy as np
from events import Events

from Model.FitnessTable.sorted_fitness_table import SortedFitnessTable
from Model.Individual.individual import Individual


class OptimizerParallelNGB:
    """Non-Generation-Based(NGB) optimization engine"""

    def __init__(self, params_ngb, selection, recombination,
                 mutation, statistics_recorder):
        self.params_ngb = params_ngb
        self.select_from = selection
        self.recombine = recombination
        self.mutate = mutation
        self.fitness_table = SortedFitnessTable()
        self.fit_calls_remain = 0
        self.lock = threading.Lock()
        self.events = Events()
        if statistics_recorder is not None:
            self.events.on_change += statistics_recorder.record

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

    def optimization_routine(self, fitness_object):
        while self.fit_calls_remain > 0:
            with self.lock:
                parents_indices = self.select_from(len(self.fitness_table))
                recombined = self.recombine([self.fitness_table[x] for x in parents_indices])
                mutated = self.mutate(recombined)

            mapped_params = self.map_parameters(mutated.params)  # map the params from range (0.0, 1.0) to search_space
            mutated.fitness = fitness_object.compute(mapped_params)

            with self.lock:
                if self.fit_calls_remain > 0:
                    self.fitness_table.add_individual(mutated)

                    if len(self.fitness_table) > self.params_ngb.table_size:
                        self.fitness_table.remove_last()

                    self.events.on_change(iteration=self.params_ngb.max_fitness_calls - self.fit_calls_remain,
                                          fitness_table=self.fitness_table)

                    self.fit_calls_remain -= 1

    def optimization_start(self, fitness_object):
        self.fitness_table.clear()
        self.table_random_initialization(fitness_object)
        self.fit_calls_remain = self.params_ngb.max_fitness_calls - self.params_ngb.initial_table_size
        self.events.on_change(iteration=0, fitness_table=self.fitness_table)

        threads = []
        for i in range(multiprocessing.cpu_count()):
            p = threading.Thread(target=self.optimization_routine, args=(fitness_object, ))
            threads.append(p)

        for p in threads:
            p.start()

        for p in threads:
            p.join()

        return SortedFitnessTable.from_table(self.fitness_table)
