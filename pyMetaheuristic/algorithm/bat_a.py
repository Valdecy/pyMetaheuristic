############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com 
# Metaheuristic: Bat Algorithm

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

# Function: Initialize Variables
def initial_position(swarm_size, dim):
    velocity  = np.zeros((swarm_size, dim))
    frequency = np.zeros((swarm_size, 1))
    rate      = np.random.rand(swarm_size, 1)  
    loudness  = np.random.uniform(1, 2, (swarm_size, 1))  
    return velocity, frequency, rate, loudness

############################################################################

# Function: Update Position
def update_position(position, velocity, frequency, rate, loudness, best_ind, alpha, gama, fmin, fmax, count, min_values, max_values, target_function):
    dim                  = len(min_values)
    position_            = np.zeros_like(position)
    beta                 = np.random.rand(position.shape[0])
    rand                 = np.random.rand(position.shape[0])
    rand_position_update = np.random.rand(position.shape[0])
    frequency[:, 0]      = fmin + (fmax - fmin) * beta
    velocity             = velocity + (position[:, :-1] - best_ind[:-1]) * frequency
    position_[:, :-1]    = np.clip(position[:, :-1] + velocity, min_values, max_values)
    for i in range(0, position.shape[0]):
        position_[i, -1] = target_function(position_[i, :-1])
        if (rand[i] > rate[i, 0]):
            loudness_mean     = loudness.mean()
            random_shift      = np.random.uniform(-1, 1, dim) * loudness_mean
            position_[i, :-1] = np.clip(best_ind[:-1] + random_shift, min_values, max_values)
            position_[i,  -1] = target_function(position_[i, :-1])
        else:
            position_[i, :]   = initial_variables(1, min_values, max_values, target_function)
        if (rand_position_update[i] < position[i, -1] and position_[i, -1] <= position[i, -1]):
            position[i, :-1]  = position_[i, :-1]
            position[i,  -1]  = position_[i,  -1]
            rate[i, 0]        = np.random.rand() * (1 - np.exp(-gama * count))
            loudness[i, 0]    = loudness[i, 0] * alpha
    position   = np.vstack([position, position_])
    position   = position[position[:, -1].argsort()]
    position   = position[:position_.shape[0], :]
    best_index = np.argmin(position[:, -1])
    if (best_ind[-1] > position[best_index, -1]):
        best_ind = np.copy(position[best_index, :])
    return position, velocity, frequency, rate, loudness, best_ind

############################################################################

# Functio: BA
def bat_algorithm(swarm_size = 50, min_values = [-5,-5], max_values = [5,5], iterations = 500, alpha = 0.9, gama = 0.9, fmin = 0, fmax = 10, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position                            = initial_variables(swarm_size, min_values, max_values, target_function, start_init)
    velocity, frequency, rate, loudness = initial_position (swarm_size, len(min_values))
    best_bat                            = np.copy(position[position[:,-1].argsort()][0,:])
    count                               = 0
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_bat[-1])       
        position, velocity, frequency, rate, loudness, best_bat = update_position(position, velocity, frequency, rate, loudness, best_bat, alpha, gama, fmin, fmax, count, min_values, max_values, target_function)
        if (target_value is not None):
            if (best_bat[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1       
    return best_bat

############################################################################
