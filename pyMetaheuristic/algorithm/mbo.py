############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Monarch Butterfly Optimization

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

# Function: Migration
def migration_operator(region1, ratio, region2, phi):
    num_rows, num_cols      = region1.shape[0], region1.shape[1] - 1
    mask                    = (np.random.rand(num_rows, num_cols) < ratio * phi).flatten()
    k_indices               = np.random.randint(0, region2.shape[0], size = num_rows * num_cols)
    region1_flattened       = region1[:, :num_cols].flatten()
    region2_flattened       = region2[:, :num_cols].flatten()
    region1_flattened[mask] = region2_flattened[k_indices[mask]]
    region1[:, :num_cols]   = region1_flattened.reshape(num_rows, num_cols)
    return region1

# Function: Adjustment
def butterfly_adjusting_operator(region2, ratio, adj_rate, best_b, walk_size, count, generations):
    num_rows, num_cols = region2.shape[0], region2.shape[1] - 1
    alpha              = walk_size / ((count + 1) ** 2)
    step_size          = int(np.ceil(np.random.exponential(2 * generations)))
    region2_flattened  = region2[:, :num_cols].flatten()
    rand_mask          = (np.random.rand(num_rows, num_cols) < ratio).flatten()
    for i in range(num_rows * num_cols):
        if (rand_mask[i]):
            region2_flattened[i] = best_b[i % num_cols]
    for i in range(num_rows):
        for j in range(num_cols):
            if not rand_mask[i * num_cols + j]:
                k = np.random.randint(0, num_rows)
                region2_flattened[i * num_cols + j] = region2[k, j]
                if (np.random.rand() > adj_rate):
                    region2_flattened[i * num_cols + j] = region2_flattened[i * num_cols + j] + alpha * (np.tan(np.pi * np.random.rand()) - 0.5) * step_size
    region2[:, :num_cols] = region2_flattened.reshape(num_rows, num_cols)
    return region2

# Function: Update Position
def update_position(division, ratio, adj_rate, walk_size, count, generations, phi, population, best_b, min_values, max_values, target_function):
    sorted_p         = np.argsort(population[:, -1])
    region1          = population[sorted_p[:division], :]
    region2          = population[sorted_p[division:], :]
    region1          = migration_operator(region1, ratio, region2, phi)
    best_            = population[sorted_p[0], :]
    best_[:-1]       = np.clip(best_[:-1], min_values, max_values) 
    best_[ -1]       = target_function(best_[:-1])
    region2          = butterfly_adjusting_operator(region2, ratio, adj_rate, best_b, walk_size, count, generations)
    population       = np.vstack((region1, region2))
    population[:,-1] = np.apply_along_axis(target_function, 1, population[:, :-1])  
    idx              = np.argmin(population[:, -1])
    if (population[idx, -1] < best_[-1]):
        best_ = np.copy(population[idx, :])
    if (best_[-1] < best_b[-1]):
        best_b = np.copy(best_)
    return population, best_b

############################################################################

# Function: MBO
def monarch_butterfly_optimization(size = 15, ratio = 5/12, phi = 1.2, adj_rate = 5/12, walk_size = 1, min_values = [-100, -100], max_values = [100, 100], generations = 1500, target_function = target_function, verbose = True, start_init = None, target_value = None):
    population = initial_variables(size, min_values, max_values, target_function, start_init)
    division   = int(np.ceil(size * 5 / 12))
    best_b     = population[np.argmin(population [:, -1]), :]
    count      = 1
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_b[-1])
        population, best_b = update_position(division, ratio, adj_rate, walk_size, count, generations, phi, population, best_b, min_values, max_values, target_function)
        if (target_value is not None):
            if (best_b[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1
    return best_b

############################################################################
