############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Population-Based Incremental Learning

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import random

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(vector, target_function = target_function):
    candidate      = np.random.uniform(vector[0], vector[1], size = (len(vector[0]),))
    fitness_values = target_function(candidate)
    candidate      = np.hstack((candidate, fitness_values))
    return candidate

############################################################################

# Function: Update Vector
def update_vector(vector, current, l_rate):
    vector[0] = vector[0] * (1.0 - l_rate) + current[0] * l_rate
    vector[1] = vector[1] * (1.0 - l_rate) + current[1] * l_rate
    return vector

# Function: Mutate Vector
def mutate_vector(vector, p_mutate, mut_factor):
    for i in range(0, len(vector[0])):
        if (random.random() < p_mutate):
            vector[0][i] = vector[0][i] * (1.0 - mut_factor) + random.uniform(0, 1) * mut_factor
            vector[1][i] = vector[1][i] * (1.0 - mut_factor) + random.uniform(0, 1) * mut_factor
    return vector

############################################################################

# Function: PBIL
def population_based_incremental_learning(size = 200, mut_factor = 0.05, l_rate = 0.1, iterations = 1500, min_values = [-100, -100], max_values  = [100, 100], target_function = target_function, verbose = True):
    p_mutate = 1.0 / len(min_values)
    vector   = [np.array(min_values) + (np.array(max_values) - np.array(min_values)) / 2, np.array(max_values)]
    best     = initial_variables(vector, target_function = target_function)
    count    = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration: ', count, ' f(x) = ', best[-1])
        current = None
        for _ in range(0, size):
            candidate = initial_variables(vector, target_function = target_function)
            if (current is None or candidate[-1] < current[-1]):
                current = np.copy(candidate)
            if (candidate[-1] < best[-1]):
                best = np.copy(candidate)
        vector = update_vector(vector, current, l_rate)
        vector = mutate_vector(vector, p_mutate, mut_factor)
        count  = count + 1
    return best

############################################################################
