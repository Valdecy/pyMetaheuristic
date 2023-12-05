############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Crow Search Algorithm

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
def update_position(population, ap, fL, min_values, max_values, target_function):
    dim            = len(min_values)
    old_population = np.copy(population)
    for i in range(0, population.shape[0]):
        idx                = np.random.choice(np.delete(np.arange(population.shape[0]), i))
        rand               = np.random.rand(dim)
        rand_i             = np.random.rand(dim)
        update_mask        = rand >= ap
        population[i, :-1] = np.where(update_mask, np.clip(population[i, :-1] + rand_i * fL * (population[idx, :-1] - population[i, :-1]), min_values, max_values), np.random.uniform(min_values, max_values))
        population[i,  -1] = target_function(population[i, :-1])
    population = np.vstack([old_population, population])
    population = population[population[:, -1].argsort()]
    population = population[:old_population.shape[0], :]
    return population

############################################################################

# Function: Crow Search Algorithm
def crow_search_algorithm(population_size = 25, ap = 0.02, fL = 0.02, min_values = [-5,-5], max_values = [5,5], iterations = 100, target_function = target_function, verbose = True, start_init = None, target_value = None):
    population = initial_variables(population_size, min_values, max_values, target_function, start_init)
    best_ind   = np.copy(population[population[:,-1].argsort()][ 0,:])
    count      = 0
    while (count <= iterations):  
        if (verbose == True):
           print('Iteration = ', count, ' f(x) = ', best_ind[-1])  
        population = update_position(population, ap, fL, min_values, max_values, target_function)
        value      = np.copy(population[0,:])
        if (best_ind[-1] > value[-1]):
            best_ind = np.copy(value)  
        if (target_value is not None):
            if (best_ind[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1 
    return best_ind

############################################################################