############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Population-Based Incremental Learning

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function, start_init = None):
    dim = len(min_values)
    if (start_init is not None):
        start_init = np.atleast_2d(start_init)
        n_rows     = size - start_init.shape[0]
        if (n_rows > 0):
            rows       = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = target_function(start_init) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, start_init)
        population     = np.hstack((start_init, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    else:
        population     = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = target_function(population) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, population)
        population     = np.hstack((population, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    return population

############################################################################

# Function: Update Vector
def update_vector(vector, current, l_rate):
    vector[0] = vector[0] * (1.0 - l_rate) + current[0] * l_rate
    vector[1] = vector[1] * (1.0 - l_rate) + current[1] * l_rate
    return vector

# Function: Mutate Vector
def mutate_vector(vector, p_mutate, mut_factor):
    for i in range(0, len(vector[0])):
        if (np.random.rand() < p_mutate):
            vector[0][i] = vector[0][i] * (1.0 - mut_factor) + np.random.uniform(0, 1) * mut_factor
            vector[1][i] = vector[1][i] * (1.0 - mut_factor) + np.random.uniform(0, 1) * mut_factor
    return vector

############################################################################

# Function: PBIL
def population_based_incremental_learning(size = 15, mut_factor = 0.05, l_rate = 0.1, iterations = 1500, min_values = [-100, -100], max_values  = [100, 100], target_function = target_function, verbose = True, start_init = None, target_value = None):
    p_mutate = 1.0 / len(min_values)
    vector   = [np.array(min_values) + (np.array(max_values) - np.array(min_values)) / 2, np.array(max_values)]
    best     = initial_variables(1, vector[0], vector[1], target_function, start_init)[0,:]
    count    = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration: ', count, ' f(x) = ', best[-1])
        current = None
        for _ in range(0, size):
            candidate = initial_variables(1, vector[0], vector[1], target_function)[0,:]
            if (current is None or candidate[-1] < current[-1]):
                current = np.copy(candidate)
            if (candidate[-1] < best[-1]):
                best = np.copy(candidate)
        vector = update_vector(vector, current, l_rate)
        vector = mutate_vector(vector, p_mutate, mut_factor)
        if (target_value is not None):
            if (best[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1
    return best

############################################################################
