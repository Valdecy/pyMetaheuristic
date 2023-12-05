############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Dispersive Flies Optimization

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

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

# Function: Update Position
def update_position(position, neighbour_best, swarm_best, min_values, max_values, dt, target_function):
    r               = np.random.rand(position.shape[0], len(min_values))
    position[:,:-1] = np.clip(neighbour_best[:-1] + r * (swarm_best[:-1] - position[:, :-1]), min_values, max_values)
    position[:, -1] = np.apply_along_axis(target_function, 1, position[:, :-1])
    r               = np.random.rand(position.shape[0])
    update_mask     = r < dt
    if (np.any(update_mask)):
        random_values             = np.random.uniform(min_values, max_values, (update_mask.sum(), len(min_values)))
        position[update_mask,:-1] = np.clip(random_values, min_values, max_values)
        position[update_mask, -1] = np.apply_along_axis(target_function, 1, position[update_mask, :-1])
    return position

############################################################################

# Function: DFO
def dispersive_fly_optimization(swarm_size = 15, min_values = [-100,-100], max_values = [100,100], generations = 500, dt = 0.2, target_function = target_function, verbose = True, start_init = None, target_value = None):
    population     = initial_variables(swarm_size, min_values, max_values, target_function, start_init)
    neighbour_best = np.copy(population[population[:,-1].argsort()][0,:])
    best_fly       = np.copy(population[population[:,-1].argsort()][0,:])
    count          = 0
    while (count <= generations):
        if (verbose == True):
            print('Generation: ', count, ' f(x) = ', best_fly[-1])
        population     = update_position(population, neighbour_best, best_fly, min_values, max_values, dt, target_function)           
        neighbour_best = np.copy(population[population[:,-1].argsort()][0,:])
        if (best_fly[-1] > neighbour_best[-1]):
           best_fly = np.copy(neighbour_best)
        if (target_value is not None):
            if (best_fly[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1 
    return best_fly

############################################################################