############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Dynamic Virtual Bats Algorithm

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

# Function: Update bats
def update_bats(population, best_bat, wave_vectors, search_points, bats, freq, v, lambda_, min_values, max_values, target_function):
    cut         = population.shape[0]
    dim         = len(min_values)
    random_mask = np.random.rand(cut) < 0.25
    if (np.any(random_mask)):
        population[random_mask,:-1] = np.random.uniform(min_values, max_values, (random_mask.sum(), dim))
        population[random_mask, -1] = np.apply_along_axis(target_function, 1, population[random_mask, :-1])
    for i in range(0, cut):
        if (random_mask[i]):
            for z in range(0, wave_vectors):
                for _ in range(0, search_points):
                    u  = -1 + 2 * np.random.rand()
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
                        v[i]               = -1 + 2 * np.random.rand()                      # Change Direction Randomly
                        lambda_[i]         = lambda_[i] + (max_values - min_values) * 0.01  # Increase Lambda
                        freq[i]            = freq[i]    - (max_values - min_values) * 0.01  # Decrease Freq
                    if (target_function(H) <= population[i, -1] and target_function(H) < best_bat[-1]):
                        population[i, :-1] = H
                        lambda_[i]         = min_values                                     # Minimize Lambda
                        freq[i]            = max_values                                     # Maximize Freq
                        v[i]               = -1 + 2 * np.random.rand()                      # Change Direction Randomly
    all_populations = np.vstack([population, population])
    all_populations = all_populations[np.argsort(all_populations[:, -1])]
    population      = all_populations[:cut, :]
    best_idx        = np.argmin(population[:, -1])
    if (population[best_idx, -1] < best_bat[-1]):
        best_bat = population[best_idx, :]
    return population, best_bat, lambda_, freq, v

############################################################################

# Function: DVBA
def dynamic_virtual_bats_algorithm(size = 20, min_values = [-100, -100], max_values = [100, 100], generations = 5000, wave_vectors = 5, search_points = 6, bats = 20, beta = 100, target_function = target_function, verbose = True, start_init = None, target_value = None):
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    population = initial_variables(size, min_values, max_values, target_function, start_init)
    freq_min   = 0.0
    freq_max   = 2.0
    rho        = (max_values - min_values) / beta
    best_bat   = np.copy(population[np.argmin(population[:, -1]), :])
    freq       = np.random.uniform(freq_min, freq_max, (size, 2)) * rho
    v          = np.random.uniform(-1, 1, (size, 2))
    lambda_    = np.random.uniform(0.01, 10.0, (size, 2)) * rho
    count      = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_bat[-1])
        population, best_bat, lambda_, freq, v = update_bats(population, best_bat, wave_vectors, search_points, bats, freq, v, lambda_, min_values, max_values, target_function)
        if (target_value is not None):
            if (best_bat[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1
    return best_bat 

############################################################################