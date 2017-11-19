import numpy as np
import Model.FitnessFunction.F101 as F101
from Helpers.mappers import map_linearly_from_to
from Helpers.statistics_recorder import StatisticsRecorder
from Model.BioMechanisms.mutation import StandardMutation
from Model.BioMechanisms.recombination import StandardRecombination
from Model.BioMechanisms.selection import RandomUniqueParentsSelection
from Model.FitnessFunction import exp_function
from Model.FitnessFunction.fitness import FitnessEvaluator
from Model.Optimizers.NonGenerationBased.optimizer_ngb import OptimizerNGB
from Model.Optimizers.NonGenerationBased.parameters_ngb import ParamsNGB

# set up parameters
search_spaces = np.array([[-600, 1000.0],
                          [-600.0, 1100.0]])

max_fitness_calls = 2
is_sigma_array = True
initial_sigma = 0.2
sigma_min = 0.0001
sigma_max = 0.35
initial_table_size = 40
max_table_size = 40
num_recombined_parents = 2
restarts_num = 1

# fitness object
fitness_object = FitnessEvaluator(F101.func)

params = ParamsNGB(search_spaces.shape[0], search_spaces, max_fitness_calls,
                   is_sigma_array, initial_sigma, initial_table_size, max_table_size)

# selection, recombination and mutation objects
selection = RandomUniqueParentsSelection(num_recombined_parents).select_random_unique_parents
recombination = StandardRecombination.recombine
mutation = StandardMutation(params.num_params, sigma_min, sigma_max).mutate

# set up optimization engine
statistics_recorder = StatisticsRecorder(1)  # or None to disable
engine = OptimizerNGB(params, selection, recombination, mutation, statistics_recorder)

optimization_results = []
for i in range(0, restarts_num):
    optimization_results.append(engine.optimization_start(fitness_object))

for i in range(0, restarts_num):
    mapped_params = map_linearly_from_to(optimization_results[i][0][0].params, [0.0, 1.0], search_spaces)
    print("fit - {0:.3f}, params - {1}, sigma - {2}".format(optimization_results[i][0][0].fitness,
                                                            mapped_params, optimization_results[i][0][0].sigma))

print(len(optimization_results[0][1].records))
print(optimization_results[0][1].records[0].get_avg_sigma())
