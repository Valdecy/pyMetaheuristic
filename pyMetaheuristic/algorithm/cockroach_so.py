############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Cockroach Swarm Optimization

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import random

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population = np.zeros((size, len(min_values)+1))
    for i in range(0, size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j])
        population[i,-1] = target_function(population[i,:-1])
    return population

############################################################################

# Function: Update Roaches
def update_roaches(population, min_values, max_values, step, joe, target_function):
    old_population  = np.copy(population)
    g_population    = np.copy(population)
    cut             = population.shape[0]
    for i in range(0, population.shape[0]):
        if (random.random() < 0.1):
            population[i,:-1] = np.random.uniform(min_values, max_values)
            population[i, -1] = target_function(population[i, :-1])
            continue
        for k in range(0, population.shape[0]):
            if (population[k, -1] < population[i, -1]):
                r                  = random.uniform(0, 1)
                population[i, :-1] = np.clip(population[i, :-1] + step * r * (population[k, :-1] - population[i, :-1]), min_values, max_values)
        for k in range(0, population.shape[0]):
            r                    = random.uniform(0, 1)
            g_population[i, :-1] = np.clip(g_population[i, :-1] + step * r * (joe[:-1] - g_population[i, :-1]), min_values, max_values)
        population[i, -1]   = target_function(population[i, :-1])
        g_population[i, -1] = target_function(g_population[i, :-1])
    population = np.vstack([np.unique(old_population, axis = 0), np.unique(g_population, axis = 0), np.unique(population, axis = 0)])
    population = population[population[:, -1].argsort()]
    population = population[:cut, :]
    idx        = np.argmin(population[:, -1])
    if (population[idx, -1] < joe[-1]):
        joe = np.copy(population[idx, :])
    return population, joe

# Function: Ruthless Behavior
def ruthless_behavior(population, min_values, max_values, joe, target_function):
    old_population = np.copy(population)
    cut            = population.shape[0]
    for i in range(0, population.shape[0]):
        for j in range(len(min_values)):
            r                = random.uniform(0, 1)
            population[i, j] = np.clip(population[i, j] + r, min_values[j], max_values[j])
        population[i, -1] = target_function(population[i, :-1])
    population = np.vstack([np.unique(old_population, axis = 0), np.unique(population, axis = 0)])
    population = population[population[:, -1].argsort()]
    population = population[:cut, :]
    idx        = np.argmin(population[:, -1])
    if (population[idx, -1] < joe[-1]):
        joe = np.copy(population[idx, :])
    return population, joe

############################################################################

# Function: Cockroach SO
def cockroach_swarm_optimization(size = 10, min_values = [-100, -100], max_values = [100, 100], generations = 5000, step = 2, target_function = target_function, verbose = True):
    population = initial_variables(size, min_values, max_values, target_function)
    idx        = np.argmin(population[:, -1])
    joe        = population[idx, :]  
    count      = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', joe[-1])
        population, joe = update_roaches(population, min_values, max_values, step, joe, target_function)
        population, joe = ruthless_behavior(population, min_values, max_values, joe, target_function)
        count           = count + 1
    return joe

############################################################################
