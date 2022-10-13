############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Dispersive Flies Optimization

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import os

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_flies(swarm_size = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((swarm_size, len(min_values)+1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             r             = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
             position[i,j] = min_values[j] + r*(max_values[j] - min_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Update Position
def update_position(position, neighbour_best, swarm_best, min_values = [-5,-5], max_values = [5,5], fly = 0, target_function = target_function):
    for j in range(0, position.shape[1] - 1):
        r                = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        position[fly, j] = np.clip((neighbour_best[j] + r*(swarm_best[j] - position[fly, j])), min_values[j], max_values[j])
    position[fly, -1] = target_function(position[fly, 0:position.shape[1]-1])
    return position

############################################################################

# DFO Function
def dispersive_fly_optimization(swarm_size = 3, min_values = [-5,-5], max_values = [5,5], generations = 50, dt = 0.2, target_function = target_function, verbose = True):
    count          = 0
    population     = initial_flies(swarm_size, min_values, max_values, target_function)
    neighbour_best = np.copy(population[population[:,-1].argsort()][0,:])
    swarm_best     = np.copy(population[population[:,-1].argsort()][0,:])
    while (count <= generations):
        if (verbose == True):
            print('Generation: ', count, ' f(x) = ', swarm_best[-1])
        for i in range (0, swarm_size):
            population = update_position(population, neighbour_best, swarm_best, min_values, max_values, i, target_function)
            r          = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (r < dt):
                for j in range(0, len(min_values)):
                    r               = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    population[i,j] = min_values[j] + r*(max_values[j] - min_values[j])
                population[i,-1] = target_function(population[i,0:population.shape[1]-1])             
        neighbour_best = np.copy(population[population[:,-1].argsort()][0,:])
        if (swarm_best[-1] > neighbour_best[-1]):
           swarm_best = np.copy(neighbour_best)
        count = count + 1
    return swarm_best

############################################################################