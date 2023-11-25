############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Hunting Search Algorithm

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

# Function: Update Hunters
def update_hunters(population, alpha, beta, c_rate, mml, idx, best_hunter, count, generations, min_radius, min_values, max_radius, max_values, target_function):
    old_population = np.copy(population)
    cut            = population.shape[0]
    epoch_iter     = generations // len(population)  
    for i in range(0, population.shape[0]):
        if (count % epoch_iter != 0):
            if random.random() < c_rate:
                d                  = random.randint(0, population.shape[0] - 1)
                population[i, :-1] = population[d, :-1]
            else:
                population[i, :-1] = population[i, :-1] + mml * (population[idx, :-1] - population[i, :-1])
            for j in range(0, len(min_values)):
                population[i, j] = population[i, j] + 2 * (random.random() - 0.5) * min_radius * (max(population[:, j]) - min(population[:, j])) * np.exp(np.log( max_radius / min_radius) * count / generations)
        elif (count % epoch_iter == 0):
            t = count // epoch_iter
            for j in range(len(min_values)):
                population[i, j] = best_hunter[j] + 2 * (random.random() - 0.5) * (max_values[j] - min_values[j]) * alpha * np.exp(-beta * t)
        population[i,:-1] = np.clip(population[i, :-1], min_values, max_values)
        population[i, -1] = target_function(population[i, :-1])
    population = np.vstack([np.unique(old_population, axis = 0), np.unique(population, axis = 0)])
    population = population[population[:, -1].argsort()]
    population = population[:cut, :]
    idx_       = np.argmin(population[:, -1])
    if (population[idx_, -1] < best_hunter[-1]):
        best_hunter = np.copy(population[idx_, :])
        idx = idx_
    return population, best_hunter, idx

############################################################################

# Function: HUS
def hunting_search_algorithm(size = 15, alpha = 0.5, beta = 0.01, mml = 0.5, c_rate = 0.5, min_radius = 0.5, max_radius = 2.0, min_values = [-100, -100], max_values = [100, 100], generations = 1500, target_function = target_function, verbose = True):
    population  = initial_variables(size, min_values, max_values, target_function)
    idx         = np.argmin(population [:, -1])
    best_hunter = population [idx, :]
    count       = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_hunter[-1])
        population, best_hunter, idx = update_hunters(population, alpha, beta, c_rate, mml, idx, best_hunter, count, generations, min_radius, min_values, max_radius, max_values, target_function)
        count = count + 1
    return best_hunter

############################################################################
