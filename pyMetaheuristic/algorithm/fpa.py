############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Flower Pollination Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import random
import os
from scipy.special import gamma

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_position(flowers = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((flowers, len(min_values)+1))
    for i in range(0, flowers):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function Levy Distribution
def levy_flight(beta = 1.5):
    beta    = beta  
    r1      = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    r2      = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    sig_num = gamma(1+beta)*np.sin((np.pi*beta)/2.0)
    sig_den = gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma   = (sig_num/sig_den)**(1/beta)
    levy    = (0.01*r1*sigma)/(abs(r2)**(1/beta))
    return levy

# Function: Global Pollination
def pollination_global(position, best_global, flower = 0, gama = 0.5, lamb = 1.4, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    x = np.copy(best_global)
    for j in range(0, len(min_values)):
        x[j] = np.clip((position[flower, j]  + gama*levy_flight(lamb)*(position[flower, j] - best_global[j])),min_values[j],max_values[j])
    x[-1]  = target_function(x[0:len(min_values)])
    return x

# Function: Local Pollination
def pollination_local(position, best_global, flower = 0, nb_flower_1 = 0, nb_flower_2 = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    x = np.copy(best_global)
    for j in range(0, len(min_values)):
        r    = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        x[j] = np.clip((position[flower, j]  + r*(position[nb_flower_1, j] - position[nb_flower_2, j])),min_values[j],max_values[j])
    x[-1]  = target_function(x[0:len(min_values)])
    return x

############################################################################

# FPA Function
def flower_pollination_algorithm(flowers = 3, min_values = [-5,-5], max_values = [5,5], iterations = 50, gama = 0.5, lamb = 1.4, p = 0.8, target_function = target_function, verbose = True):    
    count       = 0
    position    = initial_position(flowers, min_values, max_values, target_function)
    best_global = np.copy(position[position[:,-1].argsort()][0,:])
    x           = np.copy(best_global)   
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_global[-1])
        for i in range(0, position.shape[0]):
            nb_flower_1 = int(np.random.randint(position.shape[0], size = 1))
            nb_flower_2 = int(np.random.randint(position.shape[0], size = 1))
            while nb_flower_1 == nb_flower_2:
                nb_flower_1 = int(np.random.randint(position.shape[0], size = 1))
            r = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (r < p):
                x = pollination_global(position, best_global, i, gama, lamb, min_values, max_values, target_function)
            else:
                x = pollination_local(position, best_global, i, nb_flower_1, nb_flower_2, min_values, max_values, target_function)
            if (x[-1] <= position[i,-1]):
                for j in range(0, position.shape[1]):
                    position[i,j] = x[j]
            value = np.copy(position[position[:,-1].argsort()][0,:])
            if (best_global[-1] > value[-1]):
                best_global = np.copy(value) 
        count = count + 1          
    return best_global

############################################################################