############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Teaching Learning Based Optimization

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
def update_population(population, teacher, min_values, max_values, target_function):
    mean             = population.mean(axis = 0)
    teaching_factor  = np.random.choice([1, 2], size=population.shape[0])
    r_i              = np.random.rand(population.shape[0], 1)
    offspring        = np.clip(population[:, :-1] + r_i * (teacher[:-1] - teaching_factor[:, np.newaxis] * mean[:-1]), min_values, max_values)
    offspring_values = np.apply_along_axis(target_function, 1, offspring)
    offspring        = np.hstack((offspring, offspring_values[:, np.newaxis]))
    population       = np.vstack([population, offspring])
    population       = population[np.argsort(population[:, -1])]
    population       = population[:offspring.shape[0], :]
    return population

# Function: Update Learners
def update_learners(population, min_values, max_values, target_function):
    num_individuals  = population.shape[0]
    offspring        = np.zeros(population.shape)
    r_i              = np.random.rand(num_individuals, 1)
    indices          = np.array([np.random.choice(np.delete(np.arange(num_individuals), i)) for i in range(num_individuals)])
    better_random    = population[:, -1] < population[indices, -1]
    direction        = np.where(better_random, -1, 1)
    difference       = (population[:, :-1] - population[indices, :-1]) * direction[:, np.newaxis]
    updated_values   = population[:, :-1] + r_i * difference
    clipped_values   = np.clip(updated_values, min_values, max_values)
    offspring[:,:-1] = clipped_values
    offspring_values = np.apply_along_axis(target_function, 1, offspring[:, :-1])
    offspring[:, -1] = offspring_values
    population       = np.vstack([population, offspring])
    population       = population[np.argsort(population[:, -1])]
    population       = population[:num_individuals, :]
    return population

############################################################################

# Function: TLBO
def teaching_learning_based_optimization(population_size = 50, min_values = [-100,-100], max_values = [100,100], generations = 100, target_function = target_function, verbose = True, start_init = None, target_value = None):
    population = initial_variables(population_size, min_values, max_values, target_function, start_init)
    teacher    = np.copy(population[population[:,-1].argsort()][0,:])
    count      = 0
    while (count <= generations):
        if (verbose == True):
            print('Generation = ', count, ' f(x) = ', teacher[-1]) 
        population = update_population(population, teacher, min_values, max_values, target_function)
        value      = np.copy(population[population[:,-1].argsort()][0,:])
        if (teacher[-1] > value[-1]):
            teacher = np.copy(value) 
        population = update_learners(population, min_values, max_values, target_function)
        value      = np.copy(population[population[:,-1].argsort()][0,:])
        if (teacher[-1] > value[-1]):
            teacher = np.copy(value) 
        if (target_value is not None):
            if (teacher[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1  
    return teacher

############################################################################