############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Hunting Search Algorithm

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

# Function: Update Hunters
def update_hunters(population, alpha, beta, c_rate, mml, idx, best_hunter, count, generations, min_radius, min_values, max_radius, max_values, target_function):
    old_population = np.copy(population)
    cut            = population.shape[0]
    epoch_iter     = generations // len(population)  
    for i in range(0, population.shape[0]):
        if (count % epoch_iter != 0):
            if random.random() < c_rate:
                d                  = random.randint(0, population.shape[0] - 1)
                population[i, :-1] = population[d, :-1]
            else:
                population[i, :-1] = population[i, :-1] + mml * (population[idx, :-1] - population[i, :-1])
            for j in range(0, len(min_values)):
                population[i, j] = population[i, j] + 2 * (random.random() - 0.5) * min_radius * (max(population[:, j]) - min(population[:, j])) * np.exp(np.log( max_radius / min_radius) * count / generations)
        elif (count % epoch_iter == 0):
            t = count // epoch_iter
            for j in range(len(min_values)):
                population[i, j] = best_hunter[j] + 2 * (random.random() - 0.5) * (max_values[j] - min_values[j]) * alpha * np.exp(-beta * t)
        population[i,:-1] = np.clip(population[i, :-1], min_values, max_values)
        population[i, -1] = target_function(population[i, :-1])
    population = np.vstack([np.unique(old_population, axis = 0), np.unique(population, axis = 0)])
    population = population[population[:, -1].argsort()]
    population = population[:cut, :]
    idx_       = np.argmin(population[:, -1])
    if (population[idx_, -1] < best_hunter[-1]):
        best_hunter = np.copy(population[idx_, :])
        idx = idx_
    return population, best_hunter, idx

############################################################################

# Function: HUS
def hunting_search_algorithm(size = 15, alpha = 0.5, beta = 0.01, mml = 0.5, c_rate = 0.5, min_radius = 0.5, max_radius = 2.0, min_values = [-100, -100], max_values = [100, 100], generations = 1500, target_function = target_function, verbose = True, start_init = None, target_value = None):
    population  = initial_variables(size, min_values, max_values, target_function)
    idx         = np.argmin(population [:, -1])
    best_hunter = population [idx, :]
    count       = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_hunter[-1])
        population, best_hunter, idx = update_hunters(population, alpha, beta, c_rate, mml, idx, best_hunter, count, generations, min_radius, min_values, max_radius, max_values, target_function)
        if (target_value is not None):
            if (best_hunter[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1  
    return best_hunter

############################################################################
