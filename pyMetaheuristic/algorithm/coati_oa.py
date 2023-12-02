############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Coati Optimization Algorithm

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function):
    population     = np.random.uniform(min_values, max_values, (size, len(min_values)))
    fitness_values = np.apply_along_axis(target_function, 1, population)
    population     = np.hstack((population, fitness_values[:, np.newaxis]))
    return population

############################################################################

# Function: Update Position
def update_position(population, min_values, max_values, best, target_function):
    old_population = np.copy(population)
    r_population   = np.copy(population)
    e_population   = initial_variables(old_population.shape[0], old_population[:,:-1].min(axis = 0), old_population[:,:-1].max(axis = 0), target_function)
    cut            = population.shape[0]
    bt_            = initial_variables(1, min_values, max_values, target_function)
    bt_            = bt_[0,:]
    for i in range(0, population.shape[0]):
        r                  = np.random.rand(2)
        itg                = np.random.randint(low = 1, high = 3, size = 2, dtype = int)
        population[i, :-1] = np.clip(population[i, :-1] + r * (best[:-1] - itg * population[i, :-1]), min_values, max_values)
        if (r_population[i, -1] > bt_[-1]):
            r                    = np.random.rand(2)
            itg                  = np.random.randint(low = 1, high = 3, size = 2, dtype = int)
            r_population[i, :-1] = np.clip(r_population[i, :-1] + r * (bt_[:-1] - itg * r_population[i, :-1]), min_values, max_values)
        else:
            r                    = np.random.rand(2)
            r_population[i, :-1] = np.clip(r_population[i, :-1] + r * (r_population[i, :-1] - bt_[:-1]), min_values, max_values)
    population  [i,-1] = target_function(population[i, :-1])
    r_population[i,-1] = target_function(r_population[i, :-1])
    population         = np.vstack([old_population, r_population, e_population, population])
    population         = population[population[:, -1].argsort()]
    population         = population[:cut, :]
    idx                = np.argmin(population[:, -1])
    if (population[idx, -1] < best[-1]):
        best = np.copy(population[idx, :])
    return population, best

############################################################################

# Function: COA
def coati_optimization_algorithm(size = 25, min_values = [-100,-100], max_values = [100, 100], generations = 5000, target_function = target_function, verbose = True):
    population = initial_variables(size, min_values, max_values, target_function)
    idx        = np.argmin(population[:, -1])
    best       = population[idx, :]  
    count      = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best[-1])
        population, best = update_position(population, min_values, max_values, best, target_function)
        count            = count + 1
    return best

############################################################################
