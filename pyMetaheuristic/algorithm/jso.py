############################################################################

# Created by: Raiser /// Prof. Valdecy Pereira, D.Sc.
# University of Chinese Academy of Sciences (China) /// UFF - Universidade Federal Fluminense (Brazil)
# email:  github.com/mpraiser /// valdecy.pereira@gmail.com
# Metaheuristic: Jellyfish Search Optimizer

# Raiser /// PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function, eta, start_init = None):
    dim        = len(min_values)
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    if (start_init is not None):
        start_init = np.atleast_2d(start_init)
        n_rows     = size - start_init.shape[0]
        if (n_rows > 0):
            rows = np.zeros((n_rows, dim))
            x    = np.random.rand(dim)
            for j in range(0, dim):
                while (x[j] in [0.00, 0.25, 0.50, 0.75, 1.00]):
                    x[j] = np.random.rand()
            for i in range(0, n_rows):
                x          = eta * x * (1 - x)
                rows[i, :] = x * (max_values - min_values) + min_values
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = target_function(start_init) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, start_init)
        position       = np.hstack((start_init, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    else:
        position = np.zeros((size, dim + 1))
        x        = np.random.rand(dim)
        for j in range(0, dim):
            while (x[j] in [0.00, 0.25, 0.50, 0.75, 1.00]):
                x[j] = np.random.rand()
        for i in range(0, size):
            x               = eta * x * (1 - x)
            position[i,:-1] = x * (max_values - min_values) + min_values
            position[i, -1] = target_function(position[i, :-1])
    return position

############################################################################

# Function: Updtade Jellyfishes Position
def update_jellyfishes_position(position, best_position, beta, gamma, c_t, min_values, max_values, target_function):
    old_position    = np.copy(position)
    num_jellyfishes = position.shape[0]
    num_features    = position.shape[1] - 1
    min_values      = np.array(min_values)
    max_values      = np.array(max_values)
    mu              = position[:, :-1].mean(axis=0)
    if (c_t >= 0.5):
        rand_1           = np.random.rand(num_jellyfishes, num_features)
        rand_2           = np.random.rand(num_jellyfishes, num_features)
        best_mat         = np.tile(best_position[:-1], (num_jellyfishes, 1))
        position[:, :-1] = np.clip(position[:, :-1] + rand_1 * (best_mat - beta * rand_2 * mu), min_values, max_values)
    else:
        rand = np.random.rand()
        if (rand > 1 - c_t):
            rand_vector      = np.random.rand(num_jellyfishes, num_features)
            position[:, :-1] = np.clip(position[:, :-1] + gamma * rand_vector * (max_values - min_values), min_values, max_values)
        else:
            for i in range(0, num_jellyfishes):
                candidates      = [item for item in range(num_jellyfishes) if item != i]
                k               = np.random.choice(candidates)
                rand_vector     = np.random.rand(num_features)
                direction       = np.where(position[i, -1] >= position[k, -1], position[k, :-1] - position[i, :-1], position[i, :-1] - position[k, :-1])
                position[i, :-1] = np.clip(position[i, :-1] + rand_vector * direction, min_values, max_values)
    position[:, -1] = np.apply_along_axis(target_function, 1, position[:, :-1])
    position        = np.vstack([old_position, position])
    position        = position[position[:, -1].argsort()]
    position        = position[:old_position.shape[0], :]
    return position

############################################################################

# Function: Jellyfish Search Optimizer
def jellyfish_search_optimizer(jellyfishes = 25, min_values = [-100,-100], max_values = [100,100], eta = 4, beta = 3, gamma = 0.1, c_0 = 0.5, iterations = 500, target_function = target_function, verbose = True, start_init = None, target_value = None):
    eta           = np.clip(eta, 3.57, 4)
    position      = initial_variables(jellyfishes, min_values, max_values, target_function, eta, start_init)
    best_position = np.copy(position[np.argmin(position[:,-1]),:]) 
    count         = 0
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_position[-1])
        rand     = np.random.rand()
        c_t      = abs( (1 - count/iterations) * (2*rand - 1) )
        position = update_jellyfishes_position(position, best_position, beta, gamma, c_t, min_values, max_values, target_function)
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
