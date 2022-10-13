############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Arithmetic Optimization Algorithm

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
def initial_population(size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population = np.zeros((size, len(min_values) + 1))
    for i in range(0, size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j]) 
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])
    return population

############################################################################

# Function: Update Population
def update_population(population, elite, mu, moa, mop, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    e = 2.2204e-16
    p = np.copy(population)
    for i in range(0, population.shape[0]):
        for j in range(0, len(min_values)):
            r1 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            r3 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (r1 > moa and r2 > 0.5):
                p[i, j] = np.clip(elite[j] / (mop + e) * ( (max_values[j] - min_values[j]) * mu + min_values[j]), min_values[j], max_values[j])
            elif (r1 > moa and r2 <= 0.5):
                p[i, j] = np.clip(elite[j] * (  mop  ) * ( (max_values[j] - min_values[j]) * mu + min_values[j]), min_values[j], max_values[j])
            elif (r1 <= moa and r3 > 0.5):
                p[i, j] = np.clip(elite[j] - (  mop  ) * ( (max_values[j] - min_values[j]) * mu + min_values[j]), min_values[j], max_values[j])
            elif (r1 <= moa and r3 <= 0.5):
                p[i, j] = np.clip(elite[j] + (  mop  ) * ( (max_values[j] - min_values[j]) * mu + min_values[j]), min_values[j], max_values[j])
        p[i, -1] = target_function(population[i, :-1])
        if (p[i, -1] < population[i, -1]):
            population[i, :] = p[i, :]
    return population

############################################################################

# AOA Function
def arithmetic_optimization_algorithm(size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, alpha = 0.5, mu = 5, target_function = target_function, verbose = True):    
    count      = 0  
    population = initial_population(size, min_values, max_values, target_function)
    elite      = np.copy(population[population[:,-1].argsort()][0,:]) 
    while (count <= iterations): 
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', elite[-1]) 
        moa        = 0.2 + count*((1 - 0.2)/iterations)
        mop        = 1 - ( (count**(1/alpha)) / (iterations**(1/alpha)) )
        population = update_population(population, elite, mu, moa, mop, min_values, max_values, target_function)
        if (population[population[:,-1].argsort()][0,-1] < elite[-1]):
            elite = np.copy(population[population[:,-1].argsort()][0,:]) 
        count = count + 1
    return elite

############################################################################
