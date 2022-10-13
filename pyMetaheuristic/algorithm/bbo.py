############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Biogeography-Based Optimization

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
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population = np.zeros((population_size, len(min_values) + 1))
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population[i,j] = random.uniform(min_values[j], max_values[j]) 
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])
    return population

############################################################################

# Function: Migration
def migration(population, island, mu, lbd, target_function):
    for k in range(0, island.shape[0]):
        for j in range(0, island.shape[1]):
            rand_1 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (rand_1 < lbd[k]):
                rand_2 = (int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)) * sum(mu)
                select = mu[1];
                idx    = 0;
                while (rand_2  > select) and (idx < island.shape[0]-1):
                    idx    = idx + 1;
                    select = select + mu[idx]
                island[k, j] = population[idx, j]
            else:
                island[k, j] = population[k, j]
        island[k,-1] = target_function(island[k,:-1]) 
    return island
 
# Function: Mutation
def mutation(offspring, elite, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    d_mutation = 0            
    for i in range (elite, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])
        offspring[i,-1] = target_function(offspring[i,:-1])                        
    return offspring

############################################################################

# BBO Function
def biogeography_based_optimization(population_size = 5, mutation_rate = 0.1, elite = 0, min_values = [-5,-5], max_values = [5,5], eta = 1, generations = 50, target_function = target_function, verbose = True):    
    count      = 0
    mu         = [ (population_size + 1 - i) / (population_size + 1) for i in range(0, population_size) ]
    lbd        = [1 - mu[i] for i in range(0, population_size)]
    population = initial_population(population_size, min_values, max_values, target_function)
    island     = np.zeros((population.shape[0], population.shape[1]))
    elite_ind  = np.copy(population[population[:,-1].argsort()][0,:])
    while (count <= generations):  
        if (verbose == True):
            print('Generation = ', count, ' f(x) = ', elite_ind[-1])  
        island     = migration(population, island, mu, lbd, target_function)
        island     = mutation(island, elite, mutation_rate, eta, min_values, max_values, target_function)
        population = np.vstack([population, island])
        population = np.copy(population[population[:,-1].argsort()])
        population = population[0:population_size,:]
        value      = np.copy(population[0,:])
        if(elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value) 
        count = count + 1      
    return elite_ind 

############################################################################
