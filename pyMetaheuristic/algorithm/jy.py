############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Jaya (Sanskrit Word for Victory)

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

# Function: Update Position by Fitness
def update_bw_positions(position):
    best_idx       = np.argmin(position[:,-1])
    worst_idx      = np.argmax(position[:,-1])
    best_position  = position[best_idx,    :]
    worst_position = position[worst_idx,   :]
    return best_position, worst_position

# Function: Search
def update_position(position, best_position, worst_position, min_values, max_values, target_function):
    min_values          = np.array(min_values)
    max_values          = np.array(max_values)
    num_features        = position.shape[1] - 1
    a_matrix            = np.random.rand(position.shape[0], num_features)
    b_matrix            = np.random.rand(position.shape[0], num_features)
    candidate_positions = position[:, :-1] + a_matrix * (best_position[:-1] - np.abs(position[:, :-1])) - b_matrix * (worst_position[:-1] - np.abs(position[:, :-1]))
    candidate_positions = np.clip(candidate_positions, min_values, max_values)
    fitness_values      =  np.apply_along_axis(target_function, 1, candidate_positions)
    candidate_positions = np.hstack((candidate_positions, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    position            = np.vstack([position, candidate_positions])
    position            = position[position[:, -1].argsort()]
    position            = position[:candidate_positions.shape[0], :]
    return position

############################################################################

# Function: Jaya
def victory(size = 25, min_values = [-100,-100], max_values = [100,100], iterations = 5000, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position    = initial_variables(size, min_values, max_values, target_function, start_init)
    best_p      = np.copy(position[0, :])
    best_p[-1]  = float('+inf')
    worst_p     = np.copy(position[0, :])
    worst_p[-1] = 0
    count       = 0
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_p[-1])
        position        = update_position(position, best_p, worst_p, min_values, max_values, target_function)
        best_p, worst_p = update_bw_positions(position)
        if (target_value is not None):
            if (best_p[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1      
    return best_p

############################################################################
