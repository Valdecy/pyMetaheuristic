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
def initial_variables(size, min_values, max_values, target_function, start_init = None):
    dim = len(min_values)
    if (start_init is not None):
        start_init = np.atleast_2d(start_init)
        n_rows     = size - start_init.shape[0]
        if (n_rows > 0):
            rows       = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = target_function(start_init) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, start_init)
        population     = np.hstack((start_init, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    else:
        population     = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = target_function(population) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, population)
        population     = np.hstack((population, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    return population

############################################################################

# Function: Levy Distribution Vector
def levy_flight_(dimensions, beta = 1.5):
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

def levy_flight(dimensions, beta):
    r1      = np.random.rand(dimensions)
    r2      = np.random.rand(dimensions)
    sig_num = gamma(1 + beta) * np.sin(np.pi * beta / 2.0)
    sig_den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma   = (sig_num / sig_den) ** (1 / beta)
    levy    = (0.01 * r1 * sigma) / (np.abs(r2) ** (1 / beta))
    return levy

# Function: Updtade Rabbit Position by Fitness
def update_rabbit_position_(position_h, position_r):
    for i in range(0, position_h.shape[0]):
        if (position_h[i,-1] < position_r[-1]):
            position_r = np.copy(position_h[i, :])
    return position_r

def update_rabbit_position(position_h, position_r):
    idx = np.argmin(position_h[:, -1])
    if (position_h[idx, -1] < position_r[-1]):
        position_r = np.copy(position_h[idx, :])
    return position_r

# Function: Updtade Hawks Position
def update_hawks_position(position_h, position_r, e_r_factor, min_values, max_values, target_function):
    for i in range(0, position_h.shape[0]):
        escaping_energy = e_r_factor * (2*np.random.rand() -1)
        if abs(escaping_energy) >= 1:
            rand_1 = np.random.rand()
            idx    = random.choice(list(range(0, position_h.shape[1])))
            hawk   = position_h[idx, :]
            if (rand_1 < 0.5):
                a                  = np.random.rand()
                b                  = np.random.rand()
                position_h[i, :-1] = hawk[:-1] - a * abs(hawk[:-1] - 2 * b * position_h[i, :-1])
            elif (rand_1 >= 0.5):
                c                  = np.random.rand()
                d                  = np.random.rand()
                position_h[i, :-1] = (position_r[:-1] - position_h[i, :-1].mean(0)) - c * (np.asarray(max_values) - np.asarray(min_values)) * d + np.asarray(min_values)
        elif abs(escaping_energy) < 1:
            rand_2 = np.random.rand()
            if (rand_2 >= 0.5 and abs(escaping_energy) < 0.5):  # Hard Besiege
                position_h[i, :-1] = (position_r[:-1]) - escaping_energy * abs(position_r[:-1] - position_h[i, :-1])
            if (rand_2 >= 0.5 and abs(escaping_energy) >= 0.5):  # Soft Besiege 
                e                  = np.random.rand()
                jump_strength      = 2 * (1 - e)  
                position_h[i, :-1] = (position_r[:-1] - position_h[i, :-1]) - escaping_energy * abs(jump_strength * position_r[:-1] - position_h[i, :-1])
            if (rand_2 < 0.5 and abs(escaping_energy) >= 0.5):  # Soft Besiege 
                f             = np.random.rand()
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
                g             = np.random.rand()
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

# Function: HHO
def harris_hawks_optimization(hawks = 50, min_values = [-100,-100], max_values = [100,100], iterations = 2500, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position_h = initial_variables(hawks, min_values, max_values, target_function, start_init = None)
    position_r = np.copy(position_h[0, :])
    count      = 0
    while (count <= iterations): 
        if (verbose == True):    
            print('Iteration = ', count,  ' f(x) = ', position_r[-1])
        position_r =  update_rabbit_position(position_h, position_r)
        e_r_factor = 2 * (1 - (count / iterations))
        position_h = update_hawks_position(position_h, position_r, e_r_factor, min_values, max_values, target_function)
        if (target_value is not None):
            if (position_r[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1           
    return position_r

############################################################################

