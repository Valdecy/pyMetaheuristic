############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Flow Direction Algorithm

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
        position[i,-1] = target_function(position[i,:-1])
    return position

############################################################################

# Function: Create Neighbors
def neighbors(position_a, elite, beta, w_component, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    b_list = []
    for _ in range(0, beta):
        position_b = np.zeros((position_a.shape[0], len(min_values)+1))
        for i in range(0, position_a.shape[0]):
            for j in range(0, len(min_values)):
                ru              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                rn              = np.random.normal(0, 1)
                ix              = random.sample(range(0, position_a.shape[0] - 1), 1)[0]
                dt              = euclidean_distance(position_a[i, :-1], elite[:-1])
                dl              = (ru*position_a[ix, j] - ru*position_a[i,j])*dt*w_component
                position_b[i,j] = np.clip(position_a[i,j] + rn*dl, min_values[j], max_values[j])
            position_b[i,-1] = target_function(position_b[i,:-1])
        b_list.append(position_b)
    position_b = np.concatenate( b_list, axis = 0 )
    return position_b

# Function:  Elite
def elite_flow(elite, elite_a, elite_b):
    if (elite_a[-1] < elite_b[-1] and elite[-1] > elite_a[-1]):
        elite = np.copy(elite_a)
    elif (elite_b[-1] < elite_a[-1] and elite[-1] > elite_b[-1]):
        elite = np.copy(elite_b)
    return elite

# Function: Distance Calculations
def euclidean_distance(x, y):
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance       
    return distance**(1/2) 

# Function: Updtade Position
def update_position(position_a, position_b, elite, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for k in range(0, position_b.shape[0], position_a.shape[0]):
        candidate = np.copy(position_a)
        for i in range(0, position_a.shape[0]):
            for j in range(0, len(min_values)):
                if (position_b[i+k, -1] < position_a[i, -1]):
                    rn              = np.random.normal(0, 1)
                    distance        = euclidean_distance(position_a[i, :-1], position_b[i+k, :-1])
                    slope           = (position_a[i, -1] - position_b[i+k, -1])/distance
                    velocity        = rn*slope
                    candidate[i, j] = np.clip(position_a[i, j] + velocity*(position_a[i, j] - position_b[i+k, j])/distance, min_values[j], max_values[j])
                else:
                    ix = random.sample(range(0, position_a.shape[0] - 1), 1)
                    rn = np.random.normal(0, 1)
                    if (position_a[ix, -1] < position_a[i, -1]):
                        candidate[i, j] = np.clip(position_a[i, j] + rn*(position_a[ix, j] - position_a[i, j]), min_values[j], max_values[j])
                    else:
                        candidate[i, j] = np.clip(position_a[i, j] + 2*rn*(elite[j] - position_a[i, j]), min_values[j], max_values[j])
            candidate[i, -1] = target_function(candidate[i, :-1])
            if (candidate[i, -1] < position_a[i, -1]):
                position_a[i, :] = candidate[i, :]
    return position_a
    
############################################################################

# FDA Function
def flow_direction_algorithm(size = 5, beta = 8, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count      = 1
    position_a = initial_position(size, min_values, max_values, target_function)
    position_b = initial_position(size*beta, min_values, max_values, target_function)
    elite_a    = np.copy(position_a[position_a[:,-1].argsort()][0,:])
    elite_b    = np.copy(elite_a)
    elite      = np.copy(elite_a)
    while (count <= iterations):   
        if (verbose == True):    
            print('Iteration = ', count,  ' f(x) = ', elite[-1])
        r1          = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        r2          = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        w_component = ( (1 - count/iterations)**(2*r1) )*(r2*count/iterations)*r2
        elite       = elite_flow(elite, elite_a, elite_b)
        position_b  = neighbors(position_a, elite, beta, w_component, min_values, max_values, target_function)
        elite_b     = np.copy(position_b[position_b[:,-1].argsort()][0,:])
        elite       = elite_flow(elite, elite_a, elite_b)
        position_a  = update_position(position_a, position_b, elite, min_values, max_values, target_function)
        elite_a     = np.copy(position_a[position_a[:,-1].argsort()][0,:])
        elite       = elite_flow(elite, elite_a, elite_b)
        count       = count + 1           
    return elite

############################################################################
