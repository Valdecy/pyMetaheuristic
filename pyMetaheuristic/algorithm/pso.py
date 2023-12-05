############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Particle Swarm Optimization

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

# Function: Initialize Velocity
def initial_velocity(position, min_values, max_values):
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    return np.random.uniform(min_values, max_values, (position.shape[0], len(min_values)))

# Function: Individual Best
def individual_best_matrix(position, i_b_matrix):
    better_fitness_mask             = position[:, -1] < i_b_matrix[:, -1]
    i_b_matrix[better_fitness_mask] = position[better_fitness_mask]
    return i_b_matrix

# Function: Velocity
def velocity_vector(position, init_velocity, i_b_matrix, best_global, w, c1, c2):
    r1       = np.random.rand(position.shape[0], position.shape[1]-1)
    r2       = np.random.rand(position.shape[0], position.shape[1]-1)
    velocity =  w  * init_velocity + c1 * r1 * (i_b_matrix[:,:-1] - position[:,:-1]) + c2 * r2 * (best_global[ :-1] - position[:,:-1])
    return velocity

# Function: Updtade Position
def update_position(position, velocity, min_values, max_values, target_function):
    position[:,:-1] = np.clip((position[:,:-1] + velocity),  min_values,  max_values)
    position[:, -1] = np.apply_along_axis(target_function, 1, position[:,:-1])
    return position

############################################################################

# Function: PSO
def particle_swarm_optimization(swarm_size = 15, min_values = [-100,-100], max_values = [100,100], iterations = 1500, decay = 0, w = 0.9, c1 = 2, c2 = 2, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position      = initial_variables(swarm_size, min_values, max_values, target_function, start_init)
    init_velocity = initial_velocity(position, min_values, max_values)
    i_b_matrix    = np.copy(position)
    best_global   = np.copy(position[position[:,-1].argsort()][0,:])
    count         = 0
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_global[-1])
        position    = update_position(position, init_velocity, min_values, max_values, target_function)             
        i_b_matrix  = individual_best_matrix(position, i_b_matrix)
        value       = np.copy(i_b_matrix[i_b_matrix[:,-1].argsort()][0,:])
        if (best_global[-1] > value[-1]):
            best_global = np.copy(value)   
        if (decay > 0):
            n  = decay
            w  = w*(1 - ((count-1)**n)/(iterations**n))
            c1 = (1-c1)*(count/iterations) + c1
            c2 = (1-c2)*(count/iterations) + c2
        init_velocity = velocity_vector(position, init_velocity, i_b_matrix, best_global, w, c1, c2)
        if (target_value is not None):
            if (best_global[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1    
    return best_global

############################################################################
