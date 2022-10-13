############################################################################

# Created by: Raiser /// Prof. Valdecy Pereira, D.Sc.
# University of Chinese Academy of Sciences (China) /// UFF - Universidade Federal Fluminense (Brazil)
# email:  github.com/mpraiser /// valdecy.pereira@gmail.com
# Metaheuristic: Jellyfish Search Optimizer

# Raiser /// PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import random
import os

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_position(jellyfishes = 5, min_values = [-5,-5], max_values = [5,5], eta = 4, target_function = target_function):
    position = np.zeros((jellyfishes, len(min_values)+1))
    x        = []
    for j in range(0, len(min_values)):
        x_0      = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        while ([x_0] in [0.00, 0.25, 0.50, 0.75, 1.00]):
            x_0 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        x.append(x_0)
    for i in range(0, jellyfishes):
        for j in range(0, len(min_values)):
             x[j]          = eta*x[j]*(1 - x[j])
             b             = min_values[j]
             a             = max_values[j] - b
             position[i,j] = x[j]*a + b
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Updtade Jellyfishes Position
def update_jellyfishes_position(position, best_position, beta, gamma, c_t, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    mu = position.mean(axis = 0)
    if (c_t >= 0.5):
        for i in range(0, position.shape[0]):
            for j in range(0, len(min_values)):
                rand_1        = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                rand_2        = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                position[i,j] = np.clip(position[i,j] + rand_1*(best_position[0, j] - beta*rand_2*mu[j]), min_values[j], max_values[j])
            position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    else:   
        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (rand > 1 - c_t):
            for i in range(0, position.shape[0]):
                for j in range(0, len(min_values)):
                    rand          = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    position[i,j] = np.clip(position[i,j] + gamma*rand*(max_values[j] - min_values[j]), min_values[j], max_values[j])
                position[i,-1] = target_function(position[i,0:position.shape[1]-1])
        else:
            for i in range(0, position.shape[0]):
                candidates = [item for item in list(range(0, position.shape[0])) if item != i] 
                k          = random.choice(candidates)
                for j in range(0, len(min_values)):
                    rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    if (position[i, -1] >= position[k, -1]):
                        direction = position[k, j] - position[i, j]
                    else:
                        direction = position[i, j] - position[k, j]
                    position[i, j] = np.clip(position[i, j] + rand*direction, min_values[j], max_values[j])
                position[i,-1] = target_function(position[i,0:position.shape[1]-1])           
    return position

############################################################################

# Function: Jellyfish Search Optimizer
def jellyfish_search_optimizer(jellyfishes = 5, min_values = [-5,-5], max_values = [5,5], eta = 4, beta = 3, gamma = 0.1, c_0 = 0.5, iterations = 50, target_function = target_function, verbose = True):
    count         = 0
    eta           = np.clip(eta, 3.57, 4)
    position      = initial_position(jellyfishes, min_values, max_values, eta, target_function)
    best_position = np.copy(position[np.argmin(position[:,-1]),:].reshape(1,-1))   
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_position[0, -1])
        rand     = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        c_t      = abs( (1 - count/iterations) * (2*rand - 1) )
        position = update_jellyfishes_position(position, best_position, beta, gamma, c_t, min_values, max_values, target_function)
        if (np.amin(position[:,-1]) < best_position[0,-1]):
            best_position = np.copy(position[np.argmin(position[:,-1]),:].reshape(1,-1))  
        count    = count + 1
    return best_position

############################################################################
