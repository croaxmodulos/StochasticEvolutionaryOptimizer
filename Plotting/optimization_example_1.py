import numpy as np

from Model.BioMechanisms.mutation import StandardMutation
from Model.FitnessFunction import exp_function
from Model.FitnessFunction.fitness import FitnessEvaluator
from Model.Optimizers.NonGenerationBased.optimizer_ngb import OptimizerNGB
from Model.Optimizers.NonGenerationBased.parameters_ngb import ParamsNGB

# set up parameters
search_spaces = np.array([[-5.0, 10.0],
                          [-10.0, 12.0],
                          [-10.0, 12.0]])

max_fitness_calls = 6500
is_sigma_array = True
initial_sigma = 0.15
initial_table_size = 20
table_size = 40
num_recombined_parents = 2

# fitness object
fitness_object = FitnessEvaluator(exp_function.func)

params = ParamsNGB(search_spaces.shape[0], search_spaces,
                   max_fitness_calls, is_sigma_array,
                   initial_sigma, initial_table_size,
                   table_size, num_recombined_parents)

# mutation object
mutation = StandardMutation(params.tau0, params.tau1)

engine = OptimizerNGB(params, mutation)

engine.optimization_start(fitness_object)






