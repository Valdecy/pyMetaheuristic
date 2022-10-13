############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com 
# Metaheuristic: Bat Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import random
import math
import os

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_position(swarm_size = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position  = np.zeros((swarm_size, len(min_values) + 1))
    velocity  = np.zeros((swarm_size, len(min_values)))
    frequency = np.zeros((swarm_size, 1))
    rate      = np.zeros((swarm_size, 1))
    loudness  = np.zeros((swarm_size, 1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j]) 
        position[i, -1] = target_function(position[i,0:position.shape[1]-1])
        rate[i, 0]      = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        loudness[i, 0]  = random.uniform(1, 2)
    return position, velocity, frequency, rate, loudness

############################################################################

# Function: Updtade Position
def update_position(position, velocity, frequency, rate, loudness, best_ind, alpha = 0.9, gama = 0.9, fmin = 0, fmax = 10, count = 0, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position_temp = np.zeros((position.shape[0], position.shape[1]))
    for i in range(0, position.shape[0]):
        beta           = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        frequency[i,0] = fmin + (fmax - fmin)*beta
        for j in range(0, len(max_values)):
            velocity[i, j] = velocity[i,j] + (position[i,j] - best_ind[j])*frequency[i,0]
        for k in range(0, len(max_values)):
            position_temp[i,k] = position[i,k] + velocity[i,k]
            if (position_temp[i,k] > max_values[k]):
                position_temp[i,k] = max_values[k]
                velocity[i,k]      = 0
            elif(position_temp[i,k] < min_values[k]):
                position_temp[i,k] = min_values[k]
                velocity[i,k]      = 0
        position_temp[i,-1] = target_function(position_temp[i,0:len(max_values)])
        rand                = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (rand > rate[i,0]):
            for L in range(0, len(max_values)):
                position_temp[i,L] = best_ind[L] + random.uniform(-1, 1)*loudness.mean()
                if (position_temp[i,L] > max_values[L]):
                    position_temp[i,L] = max_values[L]
                    velocity[i,L]      = 0
                elif(position_temp[i,L] < min_values[L]):
                    position_temp[i,L] = min_values[L]
                    velocity[i,L]      = 0
            position_temp[i,-1] = target_function(position_temp[i,0:len(max_values)])
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (rand < position[i,-1] and position_temp[i,-1] <= position[i,-1]):
            for m in range(0, len(max_values)):
                position[i, m] = position_temp[i, m]
            position[i,-1] = target_function(position[i,0:len(max_values)])
            rate[i,0]      = rate[i,0]*(1 - math.exp(-gama*count))
            loudness[i,0]  = alpha*loudness[i,-1]
        value = np.copy(position[position[:,-1].argsort()][0,:])
        if(best_ind[-1] > value[-1]):
            best_ind = np.copy(value)
    return position, velocity, frequency, rate, loudness, best_ind

############################################################################

# BA Function
def bat_algorithm(swarm_size = 3, min_values = [-5,-5], max_values = [5,5], iterations = 50, alpha = 0.9, gama = 0.9, fmin = 0, fmax = 10, target_function = target_function, verbose = True):    
    count                                         = 0
    position, velocity, frequency, rate, loudness = initial_position(swarm_size, min_values, max_values, target_function)
    best_ind                                      = np.copy(position[position[:,-1].argsort()][0,:])
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_ind[-1])       
        position, velocity, frequency, rate, loudness, best_ind = update_position(position, velocity, frequency, rate, loudness, best_ind, alpha, gama, fmin, fmax, count, min_values, max_values, target_function)
        count                                                   = count + 1        
    return best_ind

############################################################################