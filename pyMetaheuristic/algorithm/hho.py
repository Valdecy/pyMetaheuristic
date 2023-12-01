############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Harris Hawks Optimization

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
def initial_position(hawks = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((hawks, len(min_values)+1))
    for i in range(0, hawks):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Levy Distribution Vector
def levy_flight(dimensions, beta = 1.5):
    levy = np.zeros((1, dimensions))
    for j in range(0, levy.shape[1]):
        beta       = beta  
        r1         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        r2         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        sig_num    = gamma(1+beta)*np.sin((np.pi*beta)/2.0)
        sig_den    = gamma((1+beta)/2)*beta*2**((beta-1)/2)
        sigma      = (sig_num/sig_den)**(1/beta)
        levy[0, j] = (0.01*r1*sigma)/(abs(r2)**(1/beta))
    return levy[0]

# Function: Updtade Rabbit Position by Fitness
def update_rabbit_position(position_h, position_r):
    for i in range(0, position_h.shape[0]):
        if (position_h[i,-1] < position_r[-1]):
            position_r = np.copy(position_h[i, :])
    return position_r

# Function: Updtade Hawks Position
def update_hawks_position(position_h, position_r, e_r_factor, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for i in range(0, position_h.shape[0]):
        escaping_energy = e_r_factor * (2*(int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)) -1)
        if abs(escaping_energy) >= 1:
            rand_1 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            idx    = random.choice(list(range(0, position_h.shape[1])))
            hawk   = position_h[idx, :]
            if (rand_1 < 0.5):
                a                  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                b                  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                position_h[i, :-1] = hawk[:-1] - a * abs(hawk[:-1] - 2 * b * position_h[i, :-1])
            elif (rand_1 >= 0.5):
                c                  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                d                  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                position_h[i, :-1] = (position_r[:-1] - position_h[i, :-1].mean(0)) - c * (np.asarray(max_values) - np.asarray(min_values)) * d + np.asarray(min_values)
        elif abs(escaping_energy) < 1:
            rand_2 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (rand_2 >= 0.5 and abs(escaping_energy) < 0.5):  # Hard Besiege
                position_h[i, :-1] = (position_r[:-1]) - escaping_energy * abs(position_r[:-1] - position_h[i, :-1])
            if (rand_2 >= 0.5 and abs(escaping_energy) >= 0.5):  # Soft Besiege 
                e                  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                jump_strength      = 2 * (1 - e)  
                position_h[i, :-1] = (position_r[:-1] - position_h[i, :-1]) - escaping_energy * abs(jump_strength * position_r[:-1] - position_h[i, :-1])
            if (rand_2 < 0.5 and abs(escaping_energy) >= 0.5):  # Soft Besiege 
                f             = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                jump_strength = 2 * (1 - f)
                x1            = position_r[:-1] - escaping_energy * abs(jump_strength * position_r[:-1] - position_h[i, :-1])
                for j in range (0, len(min_values)):                                 
                    x1[j] = np.clip(x1[j], min_values[j], max_values[j])
                t_x1 = target_function(x1)
                if (t_x1 < position_h[i, -1]):  
                    position_h[i, :-1] = np.copy(x1)
                    position_h[i,  -1] = t_x1
                else:  
                    x2 = position_r[:-1] - escaping_energy* abs(jump_strength * position_r[:-1] - position_h[i, :-1])+ np.multiply(np.random.randn(len(min_values)), levy_flight(len(min_values), 1.5))
                    for j in range (0, len(min_values)):                                 
                        x2[j] = np.clip(x2[j], min_values[j], max_values[j])
                    t_x2 = target_function(x2)
                    if (t_x2 < position_h[i, -1]):  
                        position_h[i, :-1] = np.copy(x2)
                        position_h[i,  -1] = t_x2
            if (rand_2 < 0.5 and abs(escaping_energy) < 0.5):  # Hard besiege 
                g             = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                jump_strength = 2 * (1 -g)
                x1            = position_r[:-1] - escaping_energy * abs(jump_strength * position_r[:-1] - position_h[i, :-1].mean(0))
                for j in range (0, len(min_values)):                                 
                    x1[j] = np.clip(x1[j], min_values[j], max_values[j])
                t_x1 = target_function(x1)
                if (t_x1 < position_h[i, -1]):  
                    position_h[i, :-1] = np.copy(x1)
                    position_h[i,  -1] = t_x1
                else:  
                    x2 = position_r[:-1] - escaping_energy * abs(jump_strength * position_r[:-1] - position_h[i, :-1].mean(0)) + np.multiply(np.random.randn(len(min_values)), levy_flight(len(min_values), 1.5))
                    for j in range (0, len(min_values)):                                 
                        x2[j] = np.clip(x2[j], min_values[j], max_values[j])
                    t_x2 = target_function(x2)
                    if (t_x2 < position_h[i, -1]):  
                        position_h[i, :-1] = np.copy(x2)
                        position_h[i,  -1] = t_x2
    return position_h

############################################################################

# HHO Function
def harris_hawks_optimization(hawks = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count      = 0
    position_h = initial_position(hawks = hawks, min_values = min_values, max_values = max_values, target_function = target_function)
    position_r = np.copy(position_h[0, :])
    while (count <= iterations): 
        if (verbose == True):    
            print('Iteration = ', count,  ' f(x) = ', position_r[-1])
        position_r =  update_rabbit_position(position_h, position_r)
        e_r_factor = 2 * (1 - (count / iterations))
        position_h = update_hawks_position(position_h, position_r, e_r_factor, min_values, max_values, target_function)
        count      = count + 1           
    return position_r

############################################################################

