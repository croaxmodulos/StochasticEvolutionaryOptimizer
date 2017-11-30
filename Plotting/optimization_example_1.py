import numpy as np
import time
import Model.FitnessFunction.F101 as F101
from Helpers.mappers import map_linearly_from_to
from Helpers.statistics_recorder import StatisticsRecorder
from Model.BioMechanisms.mutation import StandardMutation
from Model.BioMechanisms.recombination import StandardRecombination
from Model.BioMechanisms.selection import RandomUniqueParentsSelection
from Model.FitnessFunction import exp_function
from Model.FitnessFunction.fitness import FitnessEvaluator
from Model.Optimizers.NonGenerationBased.optimizer_ngb import OptimizerNGB
from Model.Optimizers.NonGenerationBased.optimizer_ngb_parallel import OptimizerParallelNGB
from Model.Optimizers.NonGenerationBased.parameters_ngb import ParamsNGB

# set up optimization parameters
search_spaces = np.array([[-600, 1000.0],
                          [-600.0, 1100.0]])
max_fitness_calls = 4000
is_sigma_array = False
initial_sigma = 0.2
sigma_min = 0.0001
sigma_max = 0.35
initial_table_size = 40
max_table_size = 40
num_recombined_parents = 2
record_after_each_iteration = 1
restarts_num = 20

# fitness object
fitness_object = FitnessEvaluator(F101.func)

params = ParamsNGB(search_spaces.shape[0], search_spaces,
                   max_fitness_calls, is_sigma_array, initial_sigma,
                   initial_table_size, max_table_size)

# set selection, recombination and mutation objects
selection = RandomUniqueParentsSelection(num_recombined_parents).select_random_unique_parents
recombination = StandardRecombination.recombine
mutation = StandardMutation(params.num_params, sigma_min, sigma_max).mutate

# set up a recorder(records fitness tables at certain iterations)
statistics_recorder = StatisticsRecorder(record_after_each_iteration)  # or None to disable

# set up optimization engine
engine = OptimizerParallelNGB(params, selection, recombination, mutation, statistics_recorder)

# launch optimizations
start = time.time()

optimization_results = []
for i in range(0, restarts_num):
    optimization_results.append(engine.optimization_start(fitness_object))

end = time.time()
# optimizations are finished

print("elapsed time {0}".format(end - start))

# print the best results in each restart
for i in range(0, restarts_num):
    mapped_params = map_linearly_from_to(optimization_results[i][0].params, [0.0, 1.0], search_spaces)
    print("fit - {0:.3f}, params - {1}, sigma - {2}".format(optimization_results[i][0].fitness,
                                                            mapped_params, optimization_results[i][0].sigma))