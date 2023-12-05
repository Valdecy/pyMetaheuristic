############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Flower Pollination Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
from scipy.special import gamma

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

# Function Levy Distribution
def levy_flight(size, beta):
    r1      = np.random.rand(size)
    r2      = np.random.rand(size)
    sig_num = gamma(1 + beta) * np.sin((np.pi * beta) / 2.0)
    sig_den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma   = (sig_num / sig_den) ** (1 / beta)
    levy    = (0.01 * r1 * sigma) / (np.abs(r2) ** (1 / beta))
    return levy

# Function: Global Pollination
def pollination_global(position, best_global, flower, gama, lamb, min_values, max_values, target_function):
    x           = np.copy(best_global)
    levy_values = levy_flight(size = len(min_values), beta=lamb)
    x[:-1]      = np.clip(position[flower, :-1] + gama * levy_values * (position[flower, :-1] - best_global[:-1]), min_values, max_values)
    x[-1]       = target_function(x[:-1])
    return x

# Function: Local Pollination
def pollination_local(position, best_global, flower, nb_flower_1, nb_flower_2, min_values, max_values, target_function):
    x      = np.copy(best_global)
    r      = np.random.rand(len(min_values))
    x[:-1] = np.clip(position[flower, :-1] + r * (position[nb_flower_1, :-1] - position[nb_flower_2, :-1]), min_values, max_values)
    x[ -1] = target_function(x[:-1])
    return x

# Function: Pollination
def pollination(position, p, best_global, gama, lamb, min_values, max_values, target_function):
    for i in range(0, position.shape[0]):
        nb_flower_1 = int(np.random.randint(position.shape[0], size = 1))
        nb_flower_2 = int(np.random.randint(position.shape[0], size = 1))
        while (nb_flower_1 == nb_flower_2):
            nb_flower_1 = int(np.random.randint(position.shape[0], size = 1))
        r = np.random.rand()
        if (r < p):
            x = pollination_global(position, best_global, i, gama, lamb, min_values, max_values, target_function)
        else:
            x = pollination_local(position, best_global, i, nb_flower_1, nb_flower_2, min_values, max_values, target_function)
        if (x[-1] <= position[i,-1]):
            for j in range(0, position.shape[1]):
                position[i,j] = x[j]
        value = np.copy(position[position[:,-1].argsort()][0,:])
        if (best_global[-1] > value[-1]):
            best_global = np.copy(value) 
    return position, best_global

############################################################################

# Function: FPA
def flower_pollination_algorithm(flowers = 25, min_values = [-100,-100], max_values = [100,100], iterations = 5000, gama = 0.5, lamb = 1.4, p = 0.8, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position    = initial_variables(flowers, min_values, max_values, target_function, start_init)
    best_global = np.copy(position[position[:,-1].argsort()][0,:])
    count       = 0   
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_global[-1])
        position, best_global = pollination(position, p, best_global, gama, lamb, min_values, max_values, target_function)
        if (target_value is not None):
            if (best_global[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1          
    return best_global

############################################################################