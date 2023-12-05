############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Cross Entropy Method

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np

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

# Function: Variables Mean
def guess_mean_calc(guess):
    guess_mean = np.mean(guess[:, :-1], axis = 0).reshape(1, -1)
    return guess_mean

# Function: Variables Standard Deviation
def guess_std_calc(guess):
    guess_std = np.std(guess[:, :-1], axis = 0).reshape(1, -1)
    return guess_std

# Function: Generate Samples
def generate_samples(guess, guess_mean, guess_std, min_values, max_values, k_samples, target_function):
    guess_sample                  = np.copy(guess[guess[:, -1].argsort()])
    new_samples                   = np.random.normal(guess_mean, guess_std, (guess.shape[0] - k_samples, len(min_values)))
    new_samples                   = np.clip(new_samples, min_values, max_values)
    guess_sample[k_samples:, :-1] = new_samples
    guess_sample[:, -1]           = np.apply_along_axis(target_function, 1, guess_sample[:, :-1])
    return guess_sample

# Function: Update Samples
def update_distribution(guess, guess_mean, guess_std, learning_rate, k_samples):
    guess_sorted                 = guess[np.argsort(guess[:, -1])]
    top_samples                  = guess_sorted[:k_samples, :-1]
    guess_mean                   = learning_rate * guess_mean + (1 - learning_rate) * np.mean(top_samples, axis = 0)
    guess_std                    = learning_rate * guess_std  + (1 - learning_rate) * np.std (top_samples, axis = 0)
    guess_std[guess_std < 0.005] = 3
    return guess_mean, guess_std

############################################################################

# Function: CEM
def cross_entropy_method(n = 5, min_values = [-5,-5], max_values = [5,5], iterations = 1000, learning_rate = 0.7, k_samples = 2, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    guess      = initial_variables(n, min_values, max_values, target_function, start_init)
    guess_mean = guess_mean_calc(guess)
    guess_std  = guess_std_calc(guess)
    best       = np.copy(guess[guess[:,-1].argsort()][0,:])
    count      = 0
    while (count < iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best[-1])
        guess                 = generate_samples(guess, guess_mean, guess_std, min_values, max_values, k_samples, target_function)
        guess_mean, guess_std = update_distribution(guess, guess_mean, guess_std, learning_rate, k_samples)
        if (best[-1] > guess[guess[:,-1].argsort()][0,:][-1]):
                best = np.copy(guess[guess[:,-1].argsort()][0,:]) 
        if (target_value is not None):
            if (best[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1     
    return best

############################################################################