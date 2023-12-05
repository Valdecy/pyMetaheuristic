############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Ant Lion Optimizer

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

# Function: Fitness Value
def fitness_calc(function_values):
    fitness_value = np.where(function_values >= 0, 1.0 / (1.0 + function_values), 1.0 + np.abs(function_values))
    return fitness_value

# Function: Fitness
def fitness_function(sources, fitness_calc):
    fitness_values     = fitness_calc(sources[:, -1])
    cumulative_sum     = np.cumsum(fitness_values)
    normalized_cum_sum = cumulative_sum / cumulative_sum[-1]
    fitness            = np.column_stack((fitness_values, normalized_cum_sum))
    return fitness

# Function: Selection
def roulette_wheel(fitness):
    ix = np.searchsorted(fitness[:, 1], np.random.rand())
    return ix

############################################################################

# Function: Random Walk
def random_walk(iterations):
    rand          = np.random.uniform(0, 1, size = iterations)
    steps         = np.where(rand > 0.5, 1, -1)
    x_random_walk = np.cumsum(steps)
    x_random_walk = np.insert(x_random_walk, 0, 0)
    return x_random_walk

# Function: Combine Ants
def combine(population, antlions):
    combination    = np.vstack([population, antlions])
    combination    = combination[combination[:, -1].argsort()]
    mid_index      = population.shape[0]
    new_population = combination[:mid_index, :]
    new_antlions   = combination[mid_index:, :]
    return new_population, new_antlions

# Function: Update Antlion
def update_ants(population, antlions, count, iterations, min_values, max_values, target_function):
    i_ratio = 1
    if (count > 0.10 * iterations):
        w_exploration = min(2 + int((count / iterations - 0.10) / 0.15), 6)
        i_ratio       = (10**w_exploration) * (count / iterations)
    sorted_antlions = antlions[antlions[:, -1].argsort()]
    elite_antlion   = sorted_antlions[ 0, :-1]
    worst_antlion   = sorted_antlions[-1, :-1]
    fitness         = fitness_function(antlions, fitness_calc)
    x_random_walk   = random_walk(iterations)
    e_random_walk   = random_walk(iterations)
    min_x, max_x    = min(x_random_walk), max(x_random_walk)
    min_e, max_e    = min(e_random_walk), max(e_random_walk)
    for i in range(0, population.shape[0]):
        ant_lion = roulette_wheel(fitness)
        for j in range(0, len(min_values)):
            rand             = np.random.rand()
            minimum_c        = (elite_antlion[j] / i_ratio) + (antlions[ant_lion, j] if rand <  0.5 else -elite_antlion[j] / i_ratio)
            maximum_d        = (worst_antlion[j] / i_ratio) + (antlions[ant_lion, j] if rand >= 0.5 else -worst_antlion[j] / i_ratio)
            x_walk           = (((x_random_walk[count] - min_x) * (maximum_d - minimum_c)) / (max_x - min_x)) + minimum_c
            e_walk           = (((e_random_walk[count] - min_e) * (maximum_d - minimum_c)) / (max_e - min_e)) + minimum_c
            population[i, j] = np.clip((x_walk + e_walk) / 2, min_values[j], max_values[j])
        population[i, -1] = target_function(population[i, :population.shape[1] - 1])
    return population, antlions

############################################################################

# ALO Function
def ant_lion_optimizer(colony_size = 500, min_values = [-5,-5], max_values = [5,5], iterations = 250, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    population = initial_variables(colony_size, min_values, max_values, target_function, start_init)
    antlions   = initial_variables(colony_size, min_values, max_values, target_function, start_init) 
    elite      = np.copy(antlions[antlions[:,-1].argsort()][0,:]) 
    count      = 0
    while (count <= iterations):
        if (verbose == True):  
            print('Iteration = ', count, ' f(x) = ', elite[-1])   
        population, antlions = update_ants(population, antlions, count, iterations, min_values, max_values, target_function)
        population, antlions = combine(population, antlions)    
        value                = np.copy(antlions[antlions[:,-1].argsort()][0,:])
        if(elite[-1] > value[-1]):
            elite = np.copy(value)
        else:
            antlions[antlions[:,-1].argsort()][0,:] = np.copy(elite)   
        if (target_value is not None):
            if (elite[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1   
    return elite

############################################################################