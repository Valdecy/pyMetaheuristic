############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Elephant Herding Optimization Algorithm

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

# Function: Update Herd
def update_herd(population, alpha, beta, best_elephant, idx_b, idx_w, min_values, max_values, target_function):
    old_population = np.copy(population)
    for i in range(0, population.shape[0]):
        for j in range(0, len(min_values)):
            if (i != idx_b and i != idx_w):
                population[i, j] = np.clip(old_population[i, j] + alpha * (best_elephant[j] - old_population[i, j]) * random.random(), min_values[j], max_values[j])
            elif (i == idx_b):
                center           = np.mean(old_population[:, j])
                population[i, j] = np.clip(beta * center, min_values[j], max_values[j])
            elif (i == idx_w):
                population[i, j] = np.clip(min_values[j] + (max_values[j] - min_values[j] + 1) * random.random(), min_values[j], max_values[j])
        population[i, -1] = target_function(population[i, :-1])
    idx_b = np.argmin(population[:, -1])
    idx_w = np.argmax(population[:, -1])
    if (population[idx_b,-1] < best_elephant[-1]):
        best_elephant = np.copy(population[idx_b, :])
    return population, best_elephant, idx_b, idx_w

############################################################################

# Function: EHO
def elephant_herding_optimization(size = 50, alpha = 0.5, beta = 0.1, min_values = [-100,-100], max_values = [100, 100], generations = 5000, target_function = target_function, verbose = True):
    population    = initial_variables(size, min_values, max_values, target_function)
    idx_b         = np.argmin(population[:, -1])
    idx_w         = np.argmax(population[:, -1])
    best_elephant = population[idx_b, :-1]
    count         = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_elephant[-1])
        population, best_elephant, idx_b, idx_w = update_herd(population, alpha, beta, best_elephant, idx_b, idx_w, min_values, max_values, target_function)
        count = count + 1
    return best_elephant

############################################################################

