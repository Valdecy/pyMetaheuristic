############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Dynamic Virtual Bats Algorithm

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

# Function: Update bats
def update_bats(population, best_bat, wave_vectors, search_points, bats, freq, v, lambda_, min_values, max_values, target_function):
    old_population = np.copy(population)
    cut            = population.shape[0]
    for i in range(0, population.shape[0]):
        if (random.random() < 0.25):
            population[i,:-1] = np.random.uniform(min_values, max_values)
            population[i, -1] = target_function(population[i, :-1])
            continue
        for z in range(0, wave_vectors):
            for _ in range(0, search_points):
                u  = -1 + 2 * random.random()
                A  = bats * u / np.clip(freq[i], 1e-9, np.inf)
                V  = v[i] + A
                V  = (V - min_values) / (max_values - min_values)
                H  = np.clip(population[i, :-1] + (z + 1) * V * lambda_[i], min_values, max_values)
                if (target_function(H) < population[i, -1]):
                    population[i, :-1] = H
                    norm               = np.clip(np.linalg.norm(H - population[i, :-1]), 1e-9, np.inf)
                    v[i]               = (H - population[i, :-1]) / norm
                    lambda_[i]         = lambda_[i] - (max_values - min_values) * 0.01  # Decrease Lambda
                    freq[i]            = freq[i]    + (max_values - min_values) * 0.01  # Increase Freq
                if (target_function(H) > best_bat[-1]):
                    v[i]               = -1 + 2 * random.random()                       # Change Direction Randomly
                    lambda_[i]         = lambda_[i] + (max_values - min_values) * 0.01  # Increase Lambda
                    freq[i]            = freq[i]    - (max_values - min_values) * 0.01  # Decrease Freq
                if (target_function(H) <= population[i, -1] and target_function(H) < best_bat[-1]):
                    population[i, :-1] = H
                    lambda_[i]         = min_values                                     # Minimize Lambda
                    freq[i]            = max_values                                     # Maximize Freq
                    v[i]               = -1 + 2 * random.random()                       # Change Direction Randomly
        population[i, -1] = target_function(population[i, :-1])
    population = np.vstack([np.unique(old_population, axis = 0), np.unique(population, axis = 0)])
    population = population[population[:, -1].argsort()]
    population = population[:cut, :]
    idx        = np.argmin(population[:, -1])
    if (population[idx, -1] < best_bat[-1]):
        best_bat = np.copy(population[idx, :])
    return population, best_bat, lambda_, freq, v

############################################################################

# Function: DVBA
def dynamic_virtual_bats_algorithm(size = 15, min_values = [-100, -100], max_values = [100, 100], generations = 5000, wave_vectors = 5, search_points = 6, bats = 20, beta = 100, target_function = target_function, verbose = True):
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    population = initial_variables(size, min_values, max_values, target_function)
    freq_min   = 0.0
    freq_max   = 2.0
    rho        = (max_values - min_values) / beta
    idx        = np.argmin(population[:, -1])
    best_bat   = np.copy(population[idx, :])
    freq       = np.random.uniform(freq_min, freq_max, (size, 2)) * rho
    v          = np.random.uniform(-1, 1, (size, 2))
    lambda_    = np.random.uniform(0.01, 10.0, (size, 2)) * rho
    count      = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_bat[-1])
        population, best_bat, lambda_, freq, v = update_bats(population, best_bat, wave_vectors, search_points, bats, freq, v, lambda_, min_values, max_values, target_function)
        count = count + 1
    return best_bat 

############################################################################