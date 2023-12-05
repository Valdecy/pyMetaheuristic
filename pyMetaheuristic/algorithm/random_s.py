############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Random Search

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

# Function: Update Position
def update_position(position, min_values, max_values, target_function):
    updated_position        = np.copy(position)
    min_values_arr          = np.array(min_values)
    max_values_arr          = np.array(max_values)
    rand_values             = np.random.uniform(0, 1, (position.shape[0], position.shape[1]-1))
    updated_position[:,:-1] = np.clip(min_values_arr + (max_values_arr - min_values_arr) * rand_values, min_values_arr, max_values_arr)
    updated_position[:, -1] = np.apply_along_axis(target_function, 1, updated_position[:, :-1])
    updated_position        = np.vstack([position,  updated_position])
    updated_position        = updated_position[np.argsort(updated_position[:, -1])]
    updated_position        = updated_position[:position.shape[0], :]
    return updated_position

############################################################################

# Function: RS
def random_search(solutions = 15, min_values = [-100,-100], max_values = [100,100], iterations = 1500, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position      = initial_variables(solutions, min_values, max_values, target_function, start_init)
    best_solution = np.copy(position[position [:,-1].argsort()][0,:])
    count         = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', best_solution[-1])        
        position = update_position(position, min_values, max_values, target_function)
        if(best_solution[-1] > position[position [:,-1].argsort()][0,:][-1]):
            best_solution = np.copy(position[position [:,-1].argsort()][0,:])      
        if (target_value is not None):
            if (best_solution[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1     
    return best_solution

############################################################################