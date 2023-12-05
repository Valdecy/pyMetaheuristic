############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Pathfinder Algorithm

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

# Function: Build Distance Matrix
def build_distance_matrix(population):
   a = np.copy(population[:,:-1])
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Update Pathfinder
def update_pathfinder(pathfinder, population, count, iterations, min_values, max_values, target_function):
    u2         = np.random.uniform(-1, 1)
    r3         = np.random.rand()
    A          = u2 * np.exp((-2 * count) / iterations)
    local_best = population[np.argmin(population[:, -1])]
    pf         = pathfinder + 2 * r3 * (pathfinder - local_best) + A
    pf[:-1]    = np.clip(pf[:-1], min_values, max_values)
    pf[ -1]    = target_function(pf[:-1])
    if (pathfinder[-1] > pf[-1]):
        pathfinder = pf.copy()
    return pathfinder

# Function: Update Position
def update_position(pathfinder, population, count, iterations, min_values, max_values, target_function):
    dist                = build_distance_matrix(population)
    u1                  = np.random.uniform(-1, 1)
    r1                  = np.random.rand(population.shape[0]**2, 1)
    r2                  = np.random.rand(population.shape[0]**2, 1)
    alpha               = np.random.uniform(1, 2, population.shape[0]**2).reshape(-1, 1)
    beta                = np.random.uniform(1, 2, population.shape[0]**2).reshape(-1, 1)
    e                   = (1 - (count / iterations)) * u1 * dist.flatten()
    pathfinder_expanded = np.tile(pathfinder[:-1], (population.shape[0]**2, 1))
    population_expanded = np.repeat(population[:, :-1], population.shape[0], axis=0)
    new_p               = population_expanded + alpha * r1 * (population_expanded - population_expanded) + beta * r2 * (pathfinder_expanded - population_expanded) + e.reshape(-1, 1)
    new_p               = np.clip(new_p, min_values, max_values)
    new_fitness         = np.apply_along_axis(target_function, 1, new_p)
    new_p               = np.hstack((new_p, new_fitness.reshape(-1, 1)))
    new_p               = np.vstack([population, new_p])
    new_p               = new_p[np.argsort(new_p[:, -1])]
    new_p               = new_p[:population.shape[0], :]
    return new_p

############################################################################

# Function: PFA
def pathfinder_algorithm(population_size = 15, min_values = [-100, -100], max_values = [100, 100], iterations = 100, target_function = target_function, verbose = True, start_init = None, target_value = None):
    count      = 0
    population = initial_variables(population_size, min_values, max_values, target_function, start_init)
    pathfinder = np.copy(population[population[:,-1].argsort()][ 0,:])
    while (count <= iterations):  
        if (verbose == True):
           print('Iteration = ', count, ' f(x) = ', pathfinder[-1])  
        population = update_position(pathfinder, population, count, iterations, min_values, max_values, target_function)
        value      = np.copy(population[population[:,-1].argsort()][ 0,:])
        if (pathfinder[-1] > value[-1]):
            pathfinder = np.copy(value)  
        pathfinder = update_pathfinder(pathfinder, population, count, iterations, min_values, max_values, target_function)
        if (target_value is not None):
            if (pathfinder[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1
    return pathfinder

############################################################################
