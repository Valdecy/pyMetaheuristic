############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Teaching Learning Based Optimization

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import random
import os

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population = np.zeros((population_size, len(min_values) + 1))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j]) 
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])
    return population

############################################################################

# Function: Update Population
def update_population(population, teacher, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    mean      = population.mean(axis = 0)
    offspring = np.zeros((population.shape))
    for i in range(0, population.shape[0]):
        teaching_factor = random.choice([1, 2])
        r_i             = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        for j in range(0, len(min_values)):
            offspring[i, j] = np.clip(population[i, j] + r_i*(teacher[j] - teaching_factor*mean[j]), min_values[j], max_values[j])
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1])
    population = np.vstack([population, offspring])
    population = population[population[:, -1].argsort()]
    population = population[:offspring.shape[0],:] 
    return population

# Function: Update Learners
def update_learners(population, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    offspring = np.zeros((population.shape))
    for i in range(0, population.shape[0]):
        r_i = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        idx = list(range(0, population.shape[0]))
        idx.remove(i)
        idx = random.choice(idx)
        for j in range(0, len(min_values)):
            if (population[i, -1] < population[idx, -1]):
                offspring[i, j] = np.clip(population[i, j] + r_i*(population[i, j] - population[idx, j]), min_values[j], max_values[j])
            else:
                offspring[i, j] = np.clip(population[i, j] + r_i*(population[idx, j] - population[i, j]), min_values[j], max_values[j])
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1])
    population = np.vstack([population, offspring])
    population = population[population[:, -1].argsort()]
    population = population[:offspring.shape[0],:] 
    return population

############################################################################

# Function: Teaching Learning Based Optimization
def teaching_learning_based_optimization(population_size = 5, min_values = [-5,-5], max_values = [5,5], generations = 100, target_function = target_function, verbose = True):
    count      = 0
    population = initial_population(population_size, min_values, max_values, target_function)
    teacher    = np.copy(population[population[:,-1].argsort()][0,:])
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
        count      = count + 1
    return teacher

############################################################################