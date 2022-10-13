############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Whale Optimization Algorithm

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
def initial_position(hunting_party = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((hunting_party, len(min_values)+1))
    for i in range(0, hunting_party):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Initialize Alpha
def leader_position(dimension = 2, target_function = target_function):
    leader = np.zeros((1, dimension+1))
    for j in range(0, dimension):
        leader[0,j] = 0.0
    leader[0,-1] = target_function(leader[0,0:leader.shape[1]-1])
    return leader

# Function: Updtade Leader by Fitness
def update_leader(position, leader):
    for i in range(0, position.shape[0]):
        if (leader[0,-1] > position[i,-1]):
            for j in range(0, position.shape[1]):
                leader[0,j] = position[i,j]
    return leader

# Function: Updtade Position
def update_position(position, leader, a_linear_component = 2, b_linear_component = 1,  spiral_param = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function): 
    for i in range(0, position.shape[0]):           
            r1_leader = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            r2_leader = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            a_leader  = 2*a_linear_component*r1_leader - a_linear_component
            c_leader  = 2*r2_leader           
            p         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)           
            for j in range (0, len(min_values)):
                if (p < 0.5):
                    if (abs(a_leader) >= 1):
                        rand              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                        rand_leader_index = math.floor(position.shape[0]*rand);
                        x_rand            = position[rand_leader_index, :]
                        distance_x_rand   = abs(c_leader*x_rand[j] - position[i,j]) 
                        position[i,j]     = np.clip( x_rand[j] - a_leader*distance_x_rand, min_values[j],  max_values[j])                                         
                    elif (abs(a_leader) < 1):
                        distance_leader = abs(c_leader*leader[0,j] - position[i,j]) 
                        position[i,j]   = np.clip(leader[0,j] - a_leader*distance_leader, min_values[j],  max_values[j])                                            
                elif (p >= 0.5):      
                    distance_Leader = abs(leader[0,j] - position[i,j])
                    rand            = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    m_param         = (b_linear_component - 1)*rand + 1
                    position[i,j]   = np.clip( (distance_Leader*math.exp(spiral_param*m_param)*math.cos(m_param*2*math.pi) + leader[0,j]), min_values[j],  max_values[j])              
            position[i,-1] = target_function(position[i,0:position.shape[1]-1])           
    return position

############################################################################

# WOA Function
def whale_optimization_algorithm(hunting_party = 5, spiral_param = 1,  min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count    = 0
    position = initial_position(hunting_party, min_values, max_values, target_function)
    leader   = leader_position(dimension = len(min_values), target_function = target_function)
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', leader[0,-1])
        a_linear_component =  2 - count*( 2/iterations)
        b_linear_component = -1 + count*(-1/iterations)
        leader             = update_leader(position, leader)
        position           = update_position(position, leader, a_linear_component, b_linear_component,  spiral_param, min_values, max_values, target_function)
        count              = count + 1   
    return leader

############################################################################
