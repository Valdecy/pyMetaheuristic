############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Cockroach Swarm Optimization

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

# Function: Update Roaches
def update_roaches(population, min_values, max_values, step, joe, target_function):
    old_population  = np.copy(population)
    g_population    = np.copy(population)
    cut             = population.shape[0]
    for i in range(0, population.shape[0]):
        if (np.random.uniform() < 0.1):
            population[i,:-1] = np.random.uniform(min_values, max_values)
            population[i, -1] = target_function(population[i, :-1])
            continue
        for k in range(0, population.shape[0]):
            if (population[k, -1] < population[i, -1]):
                r                  = np.random.uniform()
                population[i, :-1] = np.clip(population[i, :-1] + step * r * (population[k, :-1] - population[i, :-1]), min_values, max_values)
        for k in range(0, population.shape[0]):
            r                    = np.random.uniform()
            g_population[i, :-1] = np.clip(g_population[i, :-1] + step * r * (joe[:-1] - g_population[i, :-1]), min_values, max_values)
        population  [i, -1] = target_function(population[i, :-1])
        g_population[i, -1] = target_function(g_population[i, :-1])
    population = np.vstack([np.unique(old_population, axis = 0), np.unique(g_population, axis = 0), np.unique(population, axis = 0)])
    population = population[population[:, -1].argsort()]
    population = population[:cut, :]
    idx        = np.argmin(population[:, -1])
    if (population[idx, -1] < joe[-1]):
        joe = np.copy(population[idx, :])
    return population, joe

# Function: Ruthless Behavior
def ruthless_behavior(population, min_values, max_values, joe, target_function):
    cut                = population.shape[0]
    r                  = np.random.rand(cut, len(min_values))
    population[:, :-1] = np.clip(population[:, :-1] + r, min_values, max_values)
    population[:,  -1] = np.apply_along_axis(target_function, 1, population[:, :-1])
    all_populations    = np.vstack([population, population])
    all_populations    = all_populations[np.argsort(all_populations[:, -1])]
    population         = all_populations[:cut, :]
    best_idx           = np.argmin(population[:, -1])
    if (population[best_idx, -1] < joe[-1]):
        joe = population[best_idx, :]
    return population, joe

############################################################################

# Function: Cockroach SO
def cockroach_swarm_optimization(size = 10, min_values = [-100, -100], max_values = [100, 100], generations = 5000, step = 2, target_function = target_function, verbose = True, start_init = None, target_value = None):
    population = initial_variables(size, min_values, max_values, target_function, start_init)
    idx        = np.argmin(population[:, -1])
    joe        = population[idx, :]  
    count      = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', joe[-1])
        population, joe = update_roaches(population, min_values, max_values, step, joe, target_function)
        population, joe = ruthless_behavior(population, min_values, max_values, joe, target_function)
        if (target_value is not None):
            if (joe[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1 
    return joe

############################################################################
