############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Grasshopper Optimization Algorithm

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

# Function: S
def s_function(r, F, L):
    s = F*np.exp(-r/L) - np.exp(-r)
    return s

# Function: Distance Matrix
def build_distance_matrix(position):
   a = position[:,:-1]
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Update Position
def update_position(position, best_position, min_values, max_values, C, F, L, target_function):
    dim             = len(min_values)
    distance_matrix = build_distance_matrix(position)
    distance_matrix = 2 * (distance_matrix - np.min(distance_matrix)) / (np.ptp(distance_matrix) + 1e-8) + 1
    np.fill_diagonal(distance_matrix, 0)
    for j in range(dim):
        sum_grass = np.zeros(position.shape[0])
        for i in range(position.shape[0]):
            s_vals       = s_function(distance_matrix[:, i], F, L)
            denominator  = np.where(distance_matrix[:, i] == 0, 1, distance_matrix[:, i])
            sum_grass[i] = np.sum(C * ((max_values[j] - min_values[j]) / 2) * s_vals * ((position[:, j] - position[i, j]) / denominator))
        position[:, j] = np.clip(C * sum_grass + best_position[j], min_values[j], max_values[j])
    position[:, -1] = np.apply_along_axis(target_function, 1, position[:, :-1])
    return position

############################################################################

# Function: GOA
def grasshopper_optimization_algorithm(grasshoppers = 25, min_values = [-5,-5], max_values = [5,5], c_min = 0.00004, c_max = 1, iterations = 1000, F = 0.5, L = 1.5, target_function = target_function, verbose = True, start_init = None, target_value = None):
    position      = initial_variables(grasshoppers, min_values, max_values, target_function, start_init)
    best_position =  np.copy(position[np.argmin(position[:,-1]),:]) 
    count         = 0
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_position[-1])
        C        = c_max - count*( (c_max - c_min)/iterations)
        position = update_position(position, best_position, min_values, max_values, C, F, L, target_function)
        if (np.amin(position[:,-1]) < best_position[-1]):
            best_position = np.copy(position[np.argmin(position[:,-1]),:])  
        if (target_value is not None):
            if (best_position[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1 
    return best_position

############################################################################
