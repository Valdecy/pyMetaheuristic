############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Jaya (Sanskrit Word for Victory)

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

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
def initial_position(size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((size, len(min_values)+1))
    for i in range(0, size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Updtade Position by Fitness
def update_bw_positions(position, best_position, worst_position):
    for i in range(0, position.shape[0]):
        if (position[i,-1] < best_position[-1]):
            best_position = np.copy(position[i, :])
        if (position[i,-1] > worst_position[-1]):
            worst_position = np.copy(position[i, :])
    return best_position, worst_position

# Function: Search
def update_position(position, best_position, worst_position, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    candidate = np.copy(position[0, :])
    for i in range(0, position.shape[0]):
        for j in range(0, len(min_values)):
            a            = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            b            = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            candidate[j] = np.clip(position[i, j] + a * (best_position[j] - abs(position[i, j])) - b * ( worst_position[j] - abs(position[i, j])), min_values[j], max_values[j] )
        candidate[-1] = target_function(candidate[:-1])
        if (candidate[-1] < position[i,-1]):
            position[i,:] = np.copy(candidate)
    return position

############################################################################

# Jaya Function
def victory(size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count              = 0
    position           = initial_position(size, min_values, max_values, target_function)
    best_position      = np.copy(position[0, :])
    best_position[-1]  = float('+inf')
    worst_position     = np.copy(position[0, :])
    worst_position[-1] = 0
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_position[-1])
        position                      = update_position(position, best_position, worst_position, min_values, max_values, target_function)
        best_position, worst_position = update_bw_positions(position, best_position, worst_position)
        count                         = count + 1     
    return best_position 

############################################################################
