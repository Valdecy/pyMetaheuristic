############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com 
# Metaheuristic: Firefly Algorithm

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

# Function: Distance Calculations
def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y))**2))

############################################################################

# Function: Beta Value
def beta_value(x, y, gama, beta_0):
    rij  = euclidean_distance(x, y)
    beta = beta_0 * np.exp(-gama * (rij ** 2))
    return beta

# Function: Light Intensity
def light_value(light_0, x, y, gama):
    rij   = euclidean_distance(x, y)
    light = light_0 * np.exp(-gama * (rij ** 2))
    return light

# Function: Update Position
def update_position(position, alpha_0, beta_0, gama, min_values, max_values, target_function):
    dim       = len(min_values)
    position_ = np.copy(position)
    for i in range(position.shape[0]):
        for j in range(position.shape[0]):
            if (i != j):
                firefly_i = position[i, :-1]
                firefly_j = position[j, :-1]
                light_i   = light_value(position[i, -1], firefly_i, firefly_j, gama)
                light_j   = light_value(position[j, -1], firefly_i, firefly_j, gama)
                if (light_i > light_j):
                    epson           = np.random.rand(dim)
                    beta            = beta_value(firefly_i, firefly_j, gama, beta_0)
                    position[i,:-1] = np.clip(firefly_i + beta * (firefly_j - firefly_i) + alpha_0 * epson, min_values, max_values)
                    position[i, -1] = target_function(position[i, :-1])
    all_positions = np.vstack([position, position_])
    all_positions = all_positions[np.argsort(all_positions[:, -1])]
    position      = all_positions[:position_.shape[0], :]
    return position

############################################################################

# Function: FFA 
def firefly_algorithm(swarm_size = 25, min_values = [-5, -5], max_values = [5, 5], generations = 5000, alpha_0 = 0.2, beta_0 = 1, gama = 1, target_function = target_function, verbose = True, start_init = None, target_value = None):   
    position     = initial_variables(swarm_size, min_values, max_values, target_function, start_init)
    best_firefly = np.copy(position[position[:,-1].argsort()][0,:])
    count        = 0 
    while (count <= generations):
        if (verbose == True):
            print('Generation: ', count, ' f(x) = ',  best_firefly[-1])
        position     = update_position(position, alpha_0, beta_0, gama, min_values, max_values, target_function)
        best_firefly = np.copy(position[position[:,-1].argsort()][0,:])
        if (target_value is not None):
            if (best_firefly[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1  
    return best_firefly
    
############################################################################
