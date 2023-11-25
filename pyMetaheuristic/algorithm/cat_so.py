############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Cat Swarm Optimization

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
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

# Function: Update Cats
def update_cats(population, best_cat, flag, min_values, max_values, dim, dim_change, seeking_range, v, c1, target_function):
    for i in range(0, population.shape[0]):
        if (flag[i] == 0):
            copies = np.copy(population)
            for c in range(0, population.shape[0]):
                for j in range(0, len(min_values)):
                    dim[j] = 1 if j < dim_change else 0
                random.shuffle(dim)
                for j in range(0, len(min_values)):
                    if (dim[j] == 1):
                        copies[c, j] = np.clip(copies[c, j] + seeking_range * copies[c, j] * np.random.rand() if np.random.rand() > 0.5 else -seeking_range * copies[c, j] * np.random.rand(), min_values[j], max_values[j])
                copies[c, -1] = target_function(copies[c, :-1])
            idx_p = np.argmin(population[:, -1])
            idx_c = np.argmin(copies[:, -1])
            if (population[idx_p, -1] < copies[idx_c, -1]):
                cat = np.copy(population[idx_p, :])
            else:
                cat = np.copy(copies[idx_c, :])
            for j in range(0, len(min_values)):
                population[i, j] = np.clip(cat[j], min_values[j], max_values[j])
        else:
            for j in range(0, len(min_values)):
                v[i, j]          = np.clip(v[i, j] + np.random.rand() * c1 * (best_cat[j] - population[i, j]), min_values[j]*2, max_values[j]*2)
                population[i, j] = np.clip(population[i, j] + v[i, j], min_values[j], max_values[j])
        population[i, -1] = target_function(population[i, :-1])
    idx = np.argmin(population[:, -1])
    if (population[idx, -1] < best_cat[-1]):
        best_cat = np.copy(population[idx, :])
    return best_cat, population

############################################################################

# Function: Cat SO
def cat_swarm_optimization(size = 5, min_values = [-5,-5], max_values = [5,5], generations = 50, mixture_ratio = 0.2, seeking_range = 0.2, dim_change = 2, c1 = 0.5, target_function = target_function, verbose = True):
    v          = initial_variables(size, min_values, max_values, target_function)
    population = initial_variables(size, min_values, max_values, target_function) 
    idx        = np.argmin(population[:, -1])
    best_cat   = population[idx, :]
    count      = 0
    flag       = [0 for _ in range(0, size)]
    dim        = [0 for item in min_values]
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_cat[-1])
        c_sm  = int((1 - mixture_ratio) * size)
        for i in range(0, size):
            flag[i] = 0 if i < c_sm  else 1
        random.shuffle(flag)
        best_cat, population = update_cats(population, best_cat, flag, min_values, max_values, dim, dim_change, seeking_range, v, c1, target_function)
        count                = count + 1  
    return best_cat

############################################################################