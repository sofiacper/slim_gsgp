# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from slim_gsgp.initializers.initializers import rhh, grow, full
from slim_gsgp.algorithms.SLIM_GSGP.operators.crossover_operators import donor_xo, donor_n_xo, best_donor_xo, best_donor_n_xo
from slim_gsgp.algorithms.SLIM_GSGP.operators.mutators import (deflate_mutation, inflate_mutation)
from slim_gsgp.selection.selection_algorithms import tournament_selection_min, nested_tournament_selection_min
from slim_gsgp.evaluators.fitness_functions import *
from slim_gsgp.utils.utils import (get_best_min, protected_div, mean_)

# Define functions and constants

FUNCTIONS = {
    'add': {'function': torch.add, 'arity': 2},
    'subtract': {'function': torch.sub, 'arity': 2},
    'multiply': {'function': torch.mul, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2},
    #'cosine': {'function': torch.cos, 'arity':1},
    #'mean': {'function': mean_, 'arity':2}
}

CONSTANTS = {
    'constant_2': lambda _: torch.tensor(2.0),
    'constant_3': lambda _: torch.tensor(3.0),
    'constant_4': lambda _: torch.tensor(4.0),
    'constant_5': lambda _: torch.tensor(5.0),
    'constant__1': lambda _: torch.tensor(-1.0)
}

# Set parameters
settings_dict = {"p_test": 0.3}

# SLIM GSGP solve parameters
slim_gsgp_solve_parameters = {
    "run_info": None,
    "ffunction": "rmse",
    "max_depth": 17,
    "reconstruct": True,
    "n_iter": 1000,
    "elitism": True,
    "n_elites": 1,
    "log": 1,
    "verbose": 0,
    "n_jobs": 1,
    "test_elite": True
}

# SLIM GSGP parameters
slim_gsgp_parameters = {
    "initializer": "rhh",
    "selector": tournament_selection_min(2),
    "crossover": best_donor_n_xo(n = 5), #best_donor_xo(), #donor_xo,#donor_n_xo(),
    "ms": [0,1],
    "inflate_mutator": inflate_mutation,
    "deflate_mutator": deflate_mutation,
    "p_xo": 0.2,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min,
    "p_inflate": 0.3,
    "copy_parent": True,
    "operator": None,
    "pop_size": 100,
    "seed": 74,
}
slim_gsgp_parameters["p_m"] = 1 - slim_gsgp_parameters["p_xo"]

slim_gsgp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0,
    "init_depth": 6

}

fitness_function_options = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors
}

initializer_options = {
    "rhh": rhh,
    "grow": grow,
    "full": full
}