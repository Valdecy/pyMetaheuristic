############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Sine Cosine Algorithm

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

# Function: Update Position
def update_position(position, destination, r1, min_values, max_values, target_function):
    r2              = 2 * np.pi * np.random.rand(position.shape[0], len(min_values))
    r3              = 2 * np.random.rand(position.shape[0], len(min_values))
    r4              = np.random.rand(position.shape[0], len(min_values))
    sin_part        = r1 * np.sin(r2) * np.abs(r3 * destination[:-1] - position[:, :-1])
    cos_part        = r1 * np.cos(r2) * np.abs(r3 * destination[:-1] - position[:, :-1])
    position_update = np.where(r4 < 0.5, position[:, :-1] + sin_part, position[:, :-1] + cos_part)
    position[:,:-1] = np.clip(position_update, min_values, max_values)
    position[:, -1] = np.apply_along_axis(target_function, 1, position[:, :-1])
    return position

############################################################################

# Function: SCA
def sine_cosine_algorithm(solutions = 5, a_linear_component = 2,  min_values = [-100,-100], max_values = [100,100], iterations = 50, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position    = initial_variables(solutions, min_values, max_values, target_function, start_init = None)
    destination = np.copy(position[position[:,-1].argsort()][0,:])
    count       = 0
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', destination[-1])
        r1       = a_linear_component - count * (a_linear_component/iterations)   
        position = update_position(position, destination, r1 , min_values, max_values, target_function)
        value    = np.copy(position[position[:,-1].argsort()][0,:])
        if (destination[-1] > value[-1]):
            destination = np.copy(value)
        if (target_value is not None):
            if (destination[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1    
    return destination

############################################################################