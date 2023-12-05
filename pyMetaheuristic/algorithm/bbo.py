############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Biogeography-Based Optimization

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

# Function: Migration
def migration(population, island, mu, lbd, target_function):
    island_size, dim = island.shape
    rand_1           = np.random.rand(island_size)
    cum_sum_mu       = np.cumsum(mu)
    for k in range(island_size):
        migration_occurs = rand_1[k] < lbd[k]
        if migration_occurs:
            rand_2         = np.random.rand() * cum_sum_mu[-1]
            idx            = np.searchsorted(cum_sum_mu, rand_2)
            island[k, :-1] = population[idx, :-1]
        else:
            island[k, :-1] = population[k, :-1]
        island[k, -1] = target_function(island[k, :-1])
    return island

# Function: Mutation
def mutation(offspring, elite, mutation_rate, eta, min_values, max_values, target_function):
    probability            = np.random.rand(offspring.shape[0], offspring.shape[1] - 1)
    rand                   = np.random.rand(offspring.shape[0], offspring.shape[1] - 1)
    rand_d                 = np.random.rand(offspring.shape[0], offspring.shape[1] - 1)
    d_mutation             = np.where(rand <= 0.5, 2 * rand_d ** (1 / (eta + 1)) - 1, 1 - 2 * (1 - rand_d) ** (1 / (eta + 1)))
    mutation_occurs        = probability < mutation_rate
    offspring[elite:, :-1] = np.where(mutation_occurs[elite:], np.clip(offspring[elite:, :-1] + d_mutation[elite:], min_values, max_values),  offspring[elite:, :-1])
    for i in range(elite, offspring.shape[0]):
        if np.any(mutation_occurs[i, :-1]):
            offspring[i, -1] = target_function(offspring[i, :-1])
    return offspring

############################################################################

# Function: BBO
def biogeography_based_optimization(population_size = 50, mutation_rate = 0.1, elite = 1, min_values = [-5,-5], max_values = [5,5], eta = 1, generations = 500, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    mu         = [ (population_size + 1 - i) / (population_size + 1) for i in range(0, population_size) ]
    lbd        = [1 - mu[i] for i in range(0, population_size)]
    population = initial_variables(population_size, min_values, max_values, target_function, start_init)
    island     = np.zeros((population.shape[0], population.shape[1]))
    elite_ind  = np.copy(population[population[:,-1].argsort()][0,:])
    count      = 0
    while (count <= generations):  
        if (verbose == True):
            print('Generation = ', count, ' f(x) = ', elite_ind[-1])  
        island     = migration(population, island, mu, lbd, target_function)
        island     = mutation(island, elite, mutation_rate, eta, min_values, max_values, target_function)
        population = np.vstack([population, island])
        population = np.copy(population[population[:,-1].argsort()])
        population = population[0:population_size,:]
        value      = np.copy(population[0,:])
        if(elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value) 
        if (target_value is not None):
            if (elite_ind[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1        
    return elite_ind 

############################################################################
