############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Monarch Butterfly Optimization

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

# Function: Migration
def migration_operator(region1, ratio, region2, phi):
    for i in range(len(region1)):
        for j in range(len(region1[0]) - 1):
            r = phi * random.random()
            if r <= ratio:
                k = random.randint(0, len(region2) - 1)
                region1[i][j] = region2[k][j]
    return region1

# Function: Adjustment
def butterfly_adjusting_operator(region2, ratio, adj_rate, best_b, walk_size, count, generations):
    for i in range(0, len(region2)):
        alpha     = walk_size / ((count + 1) ** 2)  
        step_size = int(np.ceil(np.random.exponential(2 * generations)))
        for j in range(len(region2[0]) - 1):
            if (random.random() <= ratio):
                region2[i, j] = best_b[j]
            else:
                k             = random.randint(0, len(region2) - 1)
                region2[i, j] = region2[k, j]
                if (random.random() > adj_rate):
                    region2[i, j] += alpha * (np.tan(np.pi * random.random()) - 0.5) * step_size
    return region2

############################################################################

# Function: MBO
def monarch_butterfly_optimization(size = 30, ratio = 5/12, phi = 1.2, adj_rate = 5/12, walk_size = 1, min_values = [-100, -100], max_values = [100, 100], generations = 1500, target_function = target_function, verbose = True):
    population = initial_variables(size, min_values, max_values, target_function)
    division   = int(np.ceil(size * 5 / 12))
    idx        = np.argmin(population [:, -1])
    best_b     = population[idx, :]
    count      = 1
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_b[-1])
        sorted_p    = np.argsort(population[:, -1])
        region1     = population[sorted_p[:division], :]
        region2     = population[sorted_p[division:], :]
        region1     = migration_operator(region1, ratio, region2, phi)
        best_b      = population[sorted_p[0], :]
        best_b[:-1] = np.clip(population[sorted_p[0], :-1], min_values, max_values) 
        best_b[-1]  = target_function(best_b[:-1])
        region2     = butterfly_adjusting_operator(region2, ratio, adj_rate, best_b, walk_size, count, generations)
        population  = np.vstack((region1, region2))
        for i in range(0, population.shape[0]):
            population[i, -1] = target_function(np.clip(population[i, :-1], min_values, max_values))
        idx        = np.argmin(population[:, -1])
        if (population[idx, -1] < best_b[-1]):
            best_b = np.copy(population[idx, :])
        count     = count + 1
    return best_b

############################################################################
