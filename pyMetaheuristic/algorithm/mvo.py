############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com 
# Metaheuristic: Multi-Verse Optimizer

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
def initial_universes(universes = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    cosmos = np.zeros((universes, len(min_values) + 1))
    for i in range(0, universes):
        for j in range(0, len(min_values)):
             cosmos[i,j] = random.uniform(min_values[j], max_values[j])
        cosmos[i,-1] = target_function(cosmos[i,0:cosmos.shape[1]-1])
    return cosmos

############################################################################

# Function: Fitness
def fitness_function(cosmos): 
    fitness = np.zeros((cosmos.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ cosmos[i,-1] + abs(cosmos[:,-1].min()))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: Big Bang
def big_bang(cosmos, fitness, best_universe, wormhole_existence_probability, travelling_distance_rate, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for i in range(0,cosmos.shape[0]):
        for j in range(0,len(min_values)):
            r1 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (r1 < fitness[i, 1]):
                white_hole_i = roulette_wheel(fitness)       
                cosmos[i,j]  = cosmos[white_hole_i,j]       
            r2 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                       
            if (r2 < wormhole_existence_probability):
                r3 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
                if (r3 <= 0.5):   
                    rand        = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
                    cosmos[i,j] = best_universe [j] + travelling_distance_rate*((max_values[j] - min_values[j])*rand + min_values[j]) 
                elif (r3 > 0.5):  
                    rand        = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1) 
                    cosmos[i,j] = np.clip((best_universe [j] - travelling_distance_rate*((max_values[j] - min_values[j])*rand + min_values[j])),min_values[j],max_values[j])
        cosmos[i, -1] = target_function(cosmos[i, 0:cosmos.shape[1]-1])
    return cosmos

############################################################################

# MVO Function
def muti_verse_optimizer(universes = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count         = 0 
    cosmos        = initial_universes(universes, min_values, max_values, target_function)
    fitness       = fitness_function(cosmos)    
    best_universe = np.copy(cosmos[cosmos[:,-1].argsort()][0,:])
    wormhole_existence_probability_max = 1.0
    wormhole_existence_probability_min = 0.2   
    while (count <= iterations):  
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_universe[-1])             
        wormhole_existence_probability = wormhole_existence_probability_min + count*((wormhole_existence_probability_max - wormhole_existence_probability_min)/iterations)
        travelling_distance_rate       = 1 - (math.pow(count,1/6)/math.pow(iterations,1/6))        
        cosmos                         = big_bang(cosmos, fitness, best_universe, wormhole_existence_probability, travelling_distance_rate, min_values, max_values, target_function)
        fitness                        = fitness_function(cosmos) 
        value                          = np.copy(cosmos[cosmos[:,-1].argsort()][0,:])
        if(best_universe[-1] > value[-1]):
            best_universe  = np.copy(value)        
        count = count + 1            
    return best_universe 
 
############################################################################