############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com 
# Metaheuristic: Firefly Algorithm

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
def initial_fireflies(swarm_size = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((swarm_size, len(min_values)+1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Distance Calculations
def euclidean_distance(x, y):
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance       
    return distance**(1/2) 

############################################################################

# Function: Beta Value
def beta_value(x, y, gama = 1, beta_0 = 1):
    rij  = euclidean_distance(x, y)
    beta = beta_0*math.exp(-gama*(rij)**2)
    return beta

# Function: Ligth Intensity
def ligth_value(light_0, x, y, gama = 1):
    rij   = euclidean_distance(x, y)
    light = light_0*math.exp(-gama*(rij)**2)
    return light

# Function: Update Position
def update_position(position, x, y, alpha_0 = 0.2, beta_0 = 1, gama = 1, firefly = 0, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for j in range(0, len(x)):
        epson                = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1) - (1/2)
        position[firefly, j] = np.clip((x[j] + beta_value(x, y, gama = gama, beta_0 = beta_0)*(y[j] - x[j]) + alpha_0*epson), min_values[j], max_values[j])
    position[firefly, -1] = target_function(position[firefly, 0:position.shape[1]-1])
    return position

############################################################################

# FA Function
def firefly_algorithm(swarm_size = 3, min_values = [-5,-5], max_values = [5,5], generations = 50, alpha_0 = 0.2, beta_0 = 1, gama = 1, target_function = target_function, verbose = True):
    count    = 0    
    position = initial_fireflies(swarm_size = swarm_size, min_values = min_values, max_values = max_values, target_function = target_function)
    while (count <= generations):
        if (verbose == True):
            print('Generation: ', count, ' f(x) = ', position[position[:,-1].argsort()][0,:][-1])
        for i in range (0, swarm_size):
            for j in range(0, swarm_size):
                if (i != j):
                    firefly_i = np.copy(position[i, 0:position.shape[1]-1])
                    firefly_j = np.copy(position[j, 0:position.shape[1]-1])           
                    ligth_i   = ligth_value(position[i,-1], firefly_i, firefly_j, gama = gama)
                    ligth_j   = ligth_value(position[j,-1], firefly_i, firefly_j, gama = gama)
                    if (ligth_i > ligth_j):
                       position = update_position(position, firefly_i, firefly_j, alpha_0, beta_0, gama, i, min_values, max_values, target_function)
        count = count + 1
    best_firefly = np.copy(position[position[:,-1].argsort()][0,:])
    return best_firefly
    
############################################################################
