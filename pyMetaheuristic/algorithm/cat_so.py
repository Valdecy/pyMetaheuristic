############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Cat Swarm Optimization

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np

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

# Function: Update Cats
def update_cats(population, best_cat, flag, min_values, max_values, dim_change, seeking_range, v, c1, target_function):
    dim = len(min_values)
    for i in range(population.shape[0]):
        if (flag[i] == 0):
            dim_mask        = np.random.choice([0, 1], size = (population.shape[0], dim), p = [1 - dim_change/dim, dim_change/dim])
            seeking_changes = (2 * np.random.rand(population.shape[0], dim) - 1) * seeking_range
            copies          = np.clip(population[:,:-1] + dim_mask * seeking_changes * population[:,:-1], min_values, max_values)
            copies_fitness  = np.apply_along_axis(target_function, 1, copies)
            idx_c           = np.argmin(copies_fitness)
            best_copy       = copies[idx_c, :]
            if (copies_fitness[idx_c] < population[i, -1]):
                population[i, :-1] = best_copy[:-1]
        else:
            v[i, :-1]          = np.clip(v[i, :-1] + np.random.rand(dim) * c1 * (best_cat[:-1] - population[i, :-1]), -np.array(max_values)*2, np.array(max_values)*2)
            population[i, :-1] = np.clip(population[i, :-1] + v[i, :-1], min_values, max_values)
        population[i, -1] = target_function(population[i, :-1])
    idx = np.argmin(population[:, -1])
    if (population[idx, -1] < best_cat[-1]):
        best_cat = population[idx, :]
    return best_cat, population

############################################################################

# Function: Cat SO
def cat_swarm_optimization(size = 15, min_values = [-5,-5], max_values = [5,5], generations = 150, mixture_ratio = 0.2, seeking_range = 0.2, dim_change = 2, c1 = 0.5, target_function = target_function, verbose = True, start_init = None, target_value = None):
    v          = initial_variables(size, min_values, max_values, target_function)
    population = initial_variables(size, min_values, max_values, target_function, start_init) 
    idx        = np.argmin(population[:, -1])
    best_cat   = population[idx, :]
    count      = 0
    flag       = [0 for _ in range(0, size)]
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_cat[-1])
        c_sm  = int((1 - mixture_ratio) * size)
        for i in range(0, size):
            flag[i] = 0 if i < c_sm  else 1
        np.random.shuffle(flag)
        best_cat, population = update_cats(population, best_cat, flag, min_values, max_values, dim_change, seeking_range, v, c1, target_function)
        if (target_value is not None):
            if (best_cat[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1 
    return best_cat

############################################################################