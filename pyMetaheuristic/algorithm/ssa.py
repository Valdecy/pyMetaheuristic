############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Salp Swarm Algorithm

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
def initial_position(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Initialize Food Position
def food_position(dimension = 2, target_function = target_function):
    food = np.zeros((1, dimension+1))
    for j in range(0, dimension):
        food[0,j] = 0.0
    food[0,-1] = target_function(food[0,0:food.shape[1]-1])
    return food

# Function: Updtade Food Position by Fitness
def update_food(position, food):
    for i in range(0, position.shape[0]):
        if (food[0,-1] > position[i,-1]):
            for j in range(0, position.shape[1]):
                food[0,j] = position[i,j]
    return food

# Function: Updtade Position
def update_position(position, food, c1 = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for i in range(0, position.shape[0]):
        if (i <= position.shape[0]/2):
            for j in range (0, len(min_values)):
                c2 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                c3 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                if (c3 >= 0.5): #c3 < 0.5
                    position[i,j] = np.clip((food[0,j] + c1*((max_values[j] - min_values[j])*c2 + min_values[j])), min_values[j],max_values[j])
                else:
                    position[i,j] = np.clip((food[0,j] - c1*((max_values[j] - min_values[j])*c2 + min_values[j])), min_values[j],max_values[j])                       
        elif (i > position.shape[0]/2 and i < position.shape[0] + 1):
            for j in range (0, len(min_values)):
                position[i,j] = np.clip(((position[i - 1,j] + position[i,j])/2), min_values[j],max_values[j])             
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])         
    return position

############################################################################

# SSA Function
def salp_swarm_algorithm(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count    = 0
    position = initial_position(swarm_size, min_values, max_values, target_function)
    food     = food_position(dimension = len(min_values), target_function = target_function)
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', food[0,-1]) 
        c1       = 2*math.exp(-(4*(count/iterations))**2)
        food     = update_food(position, food)        
        position = update_position(position, food, c1, min_values, max_values, target_function)  
        count    = count + 1   
    return food

############################################################################