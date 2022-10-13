############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Sine Cosine Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import math
import random
import os

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_position(solutions = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((solutions, len(min_values)+1))
    for i in range(0, solutions):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Updtade Position
def update_position(position, destination, r1 = 2, min_values = [-5,-5], max_values = [5,5], target_function = target_function):   
    for i in range(0, position.shape[0]):
        for j in range (0, len(min_values)):         
            r2 = 2*math.pi*(int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1))
            r3 = 2*(int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1))
            r4 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)          
            if (r4 < 0.5):
                position[i,j] = np.clip((position[i,j] + (r1*math.sin(r2)*abs(r3*destination[j] - position[i,j]))),min_values[j],max_values[j])
            else:
                position[i,j] = np.clip((position[i,j] + (r1*math.cos(r2)*abs(r3*destination[j] - position[i,j]))),min_values[j],max_values[j])              
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# SCA Function
def sine_cosine_algorithm(solutions = 5, a_linear_component = 2,  min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count       = 0
    position    = initial_position(solutions, min_values, max_values, target_function)
    destination = np.copy(position[position[:,-1].argsort()][0,:])
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', destination[-1])
        r1       = a_linear_component - count*(a_linear_component/iterations)   
        position = update_position(position, destination, r1 , min_values, max_values, target_function)
        value    = np.copy(position[position[:,-1].argsort()][0,:])
        if (destination[-1] > value[-1]):
            destination = np.copy(value)
        count       = count + 1     
    return destination

############################################################################