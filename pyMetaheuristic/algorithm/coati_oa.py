############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Coati Optimization Algorithm

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

# Function: Update Position
def update_position(population, min_values, max_values, best, target_function):
    dim                = len(min_values)
    r                  = np.random.rand(population.shape[0], dim)
    itg                = np.random.randint(1, 3, size=(population.shape[0], dim))
    updated_population = np.clip(population[:, :-1] + r * (best[:-1] - itg * population[:, :-1]), min_values, max_values)
    updated_fitness    = np.apply_along_axis(target_function, 1, updated_population)
    updated_population = np.hstack((updated_population, updated_fitness.reshape(-1, 1)))
    e_population       = initial_variables(population.shape[0], min_values, max_values, target_function)
    all_populations    = np.vstack([population, updated_population, e_population])
    all_populations    = all_populations[np.argsort(all_populations[:, -1])]
    population         = all_populations[:population.shape[0], :]
    best               = population[0, :] if population[0, -1] < best[-1] else best
    return population, best

############################################################################

# Function: COA
def coati_optimization_algorithm(size = 25, min_values = [-100,-100], max_values = [100, 100], generations = 5000, target_function = target_function, verbose = True, start_init = None, target_value = None):
    population = initial_variables(size, min_values, max_values, target_function, start_init)
    idx        = np.argmin(population[:, -1])
    best       = population[idx, :]  
    count      = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best[-1])
        population, best = update_position(population, min_values, max_values, best, target_function)
        if (target_value is not None):
            if (best[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1 
    return best

############################################################################