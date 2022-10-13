############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Random Search

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
def initial_position(solutions = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((solutions, len(min_values) + 1))
    for i in range(0, solutions):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

# Function: Updtade Position
def update_position(position, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range(0, len(min_values)):
             rand                  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
             updated_position[i,j] = np.clip(min_values[j] + (max_values[j] - min_values[j])*rand, min_values[j], max_values[j])               
        updated_position[i,-1] = target_function(updated_position[i,0:updated_position.shape[1]-1])            
    return updated_position

############################################################################

# RS Function
def random_search(solutions = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count         = 0
    position      = initial_position(solutions, min_values, max_values, target_function)
    best_solution = np.copy(position[position [:,-1].argsort()][0,:])
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', best_solution[-1])        
        position = update_position(position, min_values, max_values, target_function)
        if(best_solution[-1] > position[position [:,-1].argsort()][0,:][-1]):
            best_solution = np.copy(position[position [:,-1].argsort()][0,:])      
        count = count + 1      
    return best_solution

############################################################################