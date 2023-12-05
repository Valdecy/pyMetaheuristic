############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Elephant Herding Optimization Algorithm

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

# Function: Update Herd
def update_herd(population, alpha, beta, best_elephant, idx_b, idx_w, min_values, max_values, target_function):
    old_population = np.copy(population)
    cut            = population.shape[0]
    dim            = len(min_values)
    for i in range(0, cut):
        if (i != idx_b and i != idx_w):
            r                   = np.random.rand(dim)
            population[i, :-1] = np.clip(old_population[i, :-1] + alpha * (best_elephant[:-1] - old_population[i, :-1]) * r, min_values, max_values)
        elif (i == idx_b):
            center             = np.mean(old_population[:, :-1], axis = 0)
            population[i, :-1] = np.clip(beta * center, min_values, max_values)
        elif (i == idx_w):
            random_values      = np.random.rand(dim)
            population[i, :-1] = np.clip(min_values + (max_values - min_values) * random_values, min_values, max_values)
    population[:, -1] = np.apply_along_axis(target_function, 1, population[:, :-1])
    idx_b             = np.argmin(population[:, -1])
    idx_w             = np.argmax(population[:, -1])
    if (population[idx_b, -1] < best_elephant[-1]):
        best_elephant = np.copy(population[idx_b, :])
    return population, best_elephant, idx_b, idx_w

############################################################################

# Function: EHO
def elephant_herding_optimization(size = 50, alpha = 0.5, beta = 0.1, min_values = [-100,-100], max_values = [100, 100], generations = 5000, target_function = target_function, verbose = True, start_init = None, target_value = None):
    population    = initial_variables(size, min_values, max_values, target_function, start_init)
    idx_b         = np.argmin(population[:, -1])
    idx_w         = np.argmax(population[:, -1])
    best_elephant = population[idx_b, :-1]
    min_values    = np.array(min_values)
    max_values    = np.array(max_values)
    count         = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_elephant[-1])
        population, best_elephant, idx_b, idx_w = update_herd(population, alpha, beta, best_elephant, idx_b, idx_w, min_values, max_values, target_function)
        if (target_value is not None):
            if (best_elephant[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1
    return best_elephant

############################################################################

