############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Arithmetic Optimization Algorithm

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

# Function: Update Population
def update_population(population, elite, mu, moa, mop, min_values, max_values, target_function):
    e        = 2.2204e-16
    dim      = len(min_values)
    p        = np.copy(population)
    r1       = np.random.rand(population.shape[0], dim)
    r2       = np.random.rand(population.shape[0], dim)
    r3       = np.random.rand(population.shape[0], dim)
    update_1 = np.where(r1 > moa, elite[:-1] / (mop + e) * ((max_values - min_values) * mu + min_values), elite[:-1])
    update_2 = np.where(r2 <= 0.5, update_1 * mop, update_1 - mop)
    update_3 = np.where(r3  > 0.5, update_2 - ((max_values - min_values) * mu + min_values), update_2 + ((max_values - min_values) * mu + min_values))
    up_pos   = np.clip(update_3, min_values, max_values)
    for i in range(population.shape[0]):
        new_fitness = target_function(up_pos[i, :])
        if (new_fitness < population[i, -1]):
            p[i, :-1] = up_pos[i, :]
            p[i,  -1] = new_fitness
    return p

############################################################################

# Function: AOA
def arithmetic_optimization_algorithm(size = 250, min_values = [-5,-5], max_values = [5,5], iterations = 1500, alpha = 0.5, mu = 5, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    population = initial_variables(size, min_values, max_values, target_function, start_init)
    elite      = np.copy(population[population[:,-1].argsort()][0,:]) 
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    count      = 0  
    while (count <= iterations): 
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', elite[-1]) 
        moa        = 0.2 + count*((1 - 0.2)/iterations)
        mop        = 1 - ( (count**(1/alpha)) / (iterations**(1/alpha)) )
        population = update_population(population, elite, mu, moa, mop, min_values, max_values, target_function)
        if (population[population[:,-1].argsort()][0,-1] < elite[-1]):
            elite = np.copy(population[population[:,-1].argsort()][0,:]) 
        if (target_value is not None):
            if (elite[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1
    return elite

############################################################################
