############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Salp Swarm Algorithm

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

# Function: Update Food Position by Fitness
def update_food(position, food):
    better_positions = position[position[:, -1] < food[-1]]
    if (better_positions.size > 0):
        food[:] = better_positions[np.argmin(better_positions[:, -1]), :]
    return food

# Function: Updtade Position
def update_position(position, food, c1, min_values, max_values, target_function):   
    half_size                 = position.shape[0] // 2
    range_size                = np.array(max_values) - np.array(min_values)
    c2                        = np.random.rand(position.shape[0], len(min_values))
    c3                        = np.random.rand(position.shape[0], len(min_values))
    food_matrix               = np.tile(food[:-1], (half_size, 1))
    random_matrix             = c1 * (range_size * c2[:half_size] + np.array(min_values))
    position[:half_size, :-1] = np.where(c3[:half_size] >= 0.5, np.clip(food_matrix + random_matrix, min_values, max_values), np.clip(food_matrix - random_matrix, min_values, max_values))
    if (half_size > 0 and half_size + 1 < position.shape[0]):
        avg_positions               = (position[:half_size, :-1] + position[1:half_size + 1, :-1]) / 2
        position[half_size:-1, :-1] = np.clip(avg_positions, min_values, max_values)
    position[:, -1] = np.apply_along_axis(target_function, 1, position[:, :-1])
    return position

############################################################################

# Function: SSA
def salp_swarm_algorithm(swarm_size = 5, min_values = [-100,-100], max_values = [100,100], iterations = 150, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position = initial_variables(swarm_size, min_values, max_values, target_function, start_init)
    food     = np.copy(position[np.argmin(position[:, -1]), :])
    count    = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', food[-1]) 
        c1       = 2 * np.exp(-(4 * (count/iterations))**2)
        food     = update_food(position, food)        
        position = update_position(position, food, c1, min_values, max_values, target_function)  
        if (target_value is not None):
            if (food[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1   
    return food

############################################################################