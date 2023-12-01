############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Improved Grey Wolf Optimizer

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
def initial_position(pack_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((pack_size, len(min_values)+1))
    for i in range(0, pack_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,:-1])
    return position

############################################################################

# Function: Initialize Alpha
def alpha_position(dimension = 2, target_function = target_function):
    alpha = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        alpha[0,j] = 0.0
    alpha[0,-1] = target_function(alpha[0,0:alpha.shape[1]-1])
    return alpha

# Function: Initialize Beta
def beta_position(dimension = 2, target_function = target_function):
    beta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        beta[0,j] = 0.0
    beta[0,-1] = target_function(beta[0,0:beta.shape[1]-1])
    return beta

# Function: Initialize Delta
def delta_position(dimension = 2, target_function = target_function):
    delta =  np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        delta[0,j] = 0.0
    delta[0,-1] = target_function(delta[0,0:delta.shape[1]-1])
    return delta

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i,-1] < alpha[0,-1]):
            alpha[0,:] = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[0,-1] and updated_position[i,-1] < beta[0,-1]):
            beta[0,:] = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[0,-1] and updated_position[i,-1] > beta[0,-1]  and updated_position[i,-1] < delta[0,-1]):
            delta[0,:] = np.copy(updated_position[i,:])
    return alpha, beta, delta

# Function: Updtade Position
def update_position(position, alpha, beta, delta, a_linear_component = 2, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range (0, len(min_values)):   
            r1_alpha              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            r2_alpha              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            a_alpha               = 2*a_linear_component*r1_alpha - a_linear_component
            c_alpha               = 2*r2_alpha      
            distance_alpha        = abs(c_alpha*alpha[0,j] - position[i,j]) 
            x1                    = alpha[0,j] - a_alpha*distance_alpha   
            r1_beta               = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            r2_beta               = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            a_beta                = 2*a_linear_component*r1_beta - a_linear_component
            c_beta                = 2*r2_beta            
            distance_beta         = abs(c_beta*beta[0,j] - position[i,j]) 
            x2                    = beta[0,j] - a_beta*distance_beta                          
            r1_delta              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            r2_delta              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            a_delta               = 2*a_linear_component*r1_delta - a_linear_component
            c_delta               = 2*r2_delta            
            distance_delta        = abs(c_delta*delta[0,j] - position[i,j]) 
            x3                    = delta[0,j] - a_delta*distance_delta                                 
            updated_position[i,j] = np.clip(((x1 + x2 + x3)/3), min_values[j], max_values[j])     
        updated_position[i,-1] = target_function(updated_position[i,:-1])
    return updated_position

############################################################################

# Function: Distance Calculations
def euclidean_distance(x, y):
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance       
    return distance**(1/2)

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Improve Position
def improve_position(position, updt_position, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    improve_position = np.copy(position)
    dist_matrix      = build_distance_matrix(position[:,:-1])
    for i in range(0, position.shape[0]):
        dist = euclidean_distance(position[i, :-1], updt_position[i, :-1])
        idx  = [k for k in range(0, dist_matrix[i,:].shape[0]) if dist_matrix[i, k] <= dist]
        for j in range(0, len(min_values)):
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            ix_1 = random.sample(idx, 1)[0]
            ix_2 = random.sample(range(0, position.shape[0] - 1), 1)[0]
            improve_position[i, j] = np.clip(improve_position[i, j] + rand*(position[ix_1, j] - position[ix_2, j]), min_values[j], max_values[j]) 
        improve_position[i,-1] = target_function(improve_position[i,:-1])
        min_position = min(position[i,-1], updt_position[i,-1], improve_position[i,-1])
        if (updt_position[i,-1] == min_position):
            improve_position[i,:] = np.copy(updt_position[i,:])
        elif (position[i,-1] == min_position):
            improve_position[i,:] = np.copy(position[i,:])
    return improve_position

############################################################################

# iGWO Function
def improved_grey_wolf_optimizer(pack_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count    = 0
    alpha    = alpha_position(dimension = len(min_values), target_function = target_function)
    beta     = beta_position(dimension  = len(min_values), target_function = target_function)
    delta    = delta_position(dimension = len(min_values), target_function = target_function)
    position = initial_position(pack_size, min_values, max_values, target_function)
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', alpha[0][-1])      
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        updt_position      = update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function)      
        position           = improve_position(position, updt_position, min_values, max_values, target_function)
        count              = count + 1         
    return alpha

############################################################################
