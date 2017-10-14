import numpy as np


class ParamsNGB:
    """class stores parameters for Non-Generation-Based(NGB) evolutionary optimization engine"""
    def __init__(self, num_params, search_spaces, max_fitness_calls, is_sigma_array,
                 initial_sigma, initial_table_size, table_size, num_recomb_parents):

        self.num_params = num_params  # number of parameters in the optimization
        self.search_spaces = search_spaces  # search_spaces of parameters [[a1_min, a1_max],...,[an_min, an_max]]
        self.max_fitness_calls = max_fitness_calls
        self.is_sigma_array = is_sigma_array
        self.initial_sigma = initial_sigma
        self.num_recomb_parents = num_recomb_parents
        self.initial_table_size = initial_table_size
        self.table_size = table_size
        self.tau0 = 1.0 / np.sqrt(self.num_params)  # tau0, tau1 - constants which are used in StandardMutation class
        self.tau1 = 1.0 / np.sqrt(2*np.sqrt(self.num_params))
