############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Adaptive Chaotic Grey Wolf Optimizer

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function, start_init = None, lmbda = 0.5):
    def chaotic_map(x, r):
        return np.where(x < r, x / r, (1 - x) / (1 - r))
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    dim        = len(min_values)
    if (start_init is not None):
        if (len(start_init.shape) == 1):
            start_init = start_init.reshape(1, -1)
        if (start_init.shape[0] != size):
            difference = size - start_init.shape[0]
            rows       = chaotic_map(np.random.uniform(0, 1, (abs(difference), dim)), lmbda)
            rows       = min_values + rows * (max_values - min_values)
            start_init = np.vstack((start_init[:, :dim], rows)) if difference > 0 else start_init[:size, :dim]
        fitness_values = np.apply_along_axis(target_function, 1, start_init)
        population     = np.hstack((start_init, fitness_values[:, np.newaxis]))
    else:
        random_values  = np.random.uniform(0, 1, (size, dim))
        chaotic_values = chaotic_map(random_values, lmbda)
        population     = min_values + chaotic_values * (max_values - min_values)
        population     = np.clip(population, min_values, max_values)
        fitness_values = np.apply_along_axis(target_function, 1, population)
        population     = np.hstack((population, fitness_values[:, np.newaxis]))
    return population

############################################################################

# Function: Initialize Alpha
def alpha_position(min_values, max_values, target_function):
    alpha       = np.zeros((1, len(min_values) + 1))
    alpha[0,-1] = target_function(np.clip(alpha[0,0:alpha.shape[1]-1], min_values, max_values))
    return alpha[0,:]

# Function: Initialize Beta
def beta_position(min_values, max_values, target_function):
    beta       = np.zeros((1, len(min_values) + 1))
    beta[0,-1] = target_function(np.clip(beta[0,0:beta.shape[1]-1], min_values, max_values))
    return beta[0,:]

# Function: Initialize Delta
def delta_position(min_values, max_values, target_function):
    delta       =  np.zeros((1, len(min_values) + 1))
    delta[0,-1] = target_function(np.clip(delta[0,0:delta.shape[1]-1], min_values, max_values))
    return delta[0,:]

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    idx   = np.argsort(position[:, -1])
    alpha = position[idx[0], :]
    beta  = position[idx[1], :] if position.shape[0] > 1 else alpha
    delta = position[idx[2], :] if position.shape[0] > 2 else beta
    return alpha, beta, delta

# Function: Update Position
def update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function):
    dim                     = len(min_values)
    alpha_position          = np.copy(position)
    beta_position           = np.copy(position)
    delta_position          = np.copy(position)
    updated_position        = np.copy(position)
    r1                      = np.random.rand(position.shape[0], dim)
    r2                      = np.random.rand(position.shape[0], dim)
    a                       = 2 * a_linear_component * r1 - a_linear_component
    c                       = 2 * r2
    distance_alpha          = np.abs(c * alpha[:dim] - position[:, :dim])
    distance_beta           = np.abs(c * beta [:dim] - position[:, :dim])
    distance_delta          = np.abs(c * delta[:dim] - position[:, :dim])
    x1                      = alpha[:dim] - a * distance_alpha
    x2                      = beta [:dim] - a * distance_beta
    x3                      = delta[:dim] - a * distance_delta
    alpha_position[:,:-1]   = np.clip(x1, min_values, max_values)
    beta_position [:,:-1]   = np.clip(x2, min_values, max_values)
    delta_position[:,:-1]   = np.clip(x3, min_values, max_values)
    alpha_position[:, -1]   = np.apply_along_axis(target_function, 1, alpha_position[:, :-1])
    beta_position [:, -1]   = np.apply_along_axis(target_function, 1, beta_position [:, :-1])
    delta_position[:, -1]   = np.apply_along_axis(target_function, 1, delta_position[:, :-1])
    w_alpha                 = alpha[-1] / (beta[-1] + 1e-16)
    w_beta                  = beta[-1] / (delta[-1] + 1e-16)
    updated_position[:,:-1] = np.clip((alpha_position[:, :-1] * w_alpha + beta_position[:, :-1] * w_beta + delta_position[:, :-1]) / 3, min_values, max_values)
    updated_position[:, -1] = np.apply_along_axis(target_function, 1, updated_position[:, :-1])
    updated_position        = np.vstack([position, updated_position, alpha_position, beta_position, delta_position])
    updated_position        = updated_position[updated_position[:, -1].argsort()]
    updated_position        = updated_position[:position.shape[0], :]
    return updated_position

############################################################################

# Function: ACGWO
def adaptive_chaotic_grey_wolf_optimizer(size = 15, lmbda = 0.5, min_values = [-100, -100], max_values = [100, 100], iterations = 1500, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    alpha    = alpha_position(min_values, max_values, target_function)
    beta     = beta_position (min_values, max_values, target_function)
    delta    = delta_position(min_values, max_values, target_function)
    position = initial_variables(size, min_values, max_values, target_function, start_init, lmbda)
    count    = 0
    while (count <= iterations): 
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', alpha[-1])      
        a_linear_component = 2 - 2 * ( (np.exp(count/iterations) - 1) / (np.exp(1) - 1) )
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position           = update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function = target_function)     
        if (target_value is not None):
            if (alpha[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1
    return alpha

############################################################################
