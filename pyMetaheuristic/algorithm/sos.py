############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Symbiotic Organisms Search

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

# Function: Mutation
def mutation(population, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    d_mutation = 0            
    for i in range (0, population.shape[0]):
        for j in range(0, population.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                population[i,j] = np.clip((population[i,j] + d_mutation), min_values[j], max_values[j])
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])                        
    return population

############################################################################

# Function: Mutualism Phase
def mutualism(population, elite_ind, target_function = target_function):
    x_list       = [i for i in range(0, population.shape[0])]
    population_i = np.copy(population)
    population_j = np.copy(population)
    for i in range(0, population.shape[0]):
        x_list.remove(i)
        j  = random.choice(x_list)
        r1 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        r2 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        mv = (population[i,:] + population[j,:])/2
        b1 = random.choice([1,2])
        b2 = random.choice([1,2])
        population_i[i,:]  = population[i,:] + r1*(elite_ind - mv*b1)
        population_j[i,:]  = population[j,:] + r2*(elite_ind - mv*b2)
        population_i[i,-1] = target_function(population_i[i,0:population_i.shape[1]-1])
        population_j[i,-1] = target_function(population_j[i,0:population_j.shape[1]-1])
        x_list.append(i)
    population = np.vstack([population, population_i, population_j])
    population = population[population[:, -1].argsort()]
    population = population[:population_i.shape[0],:] 
    return population

# Function: Comensalism Phase
def comensalism(population, elite_ind, target_function = target_function):
    x_list       = [i for i in range(0, population.shape[0])]
    population_i = np.copy(population)
    for i in range(0, population.shape[0]):
        x_list.remove(i)
        j = random.choice(x_list)
        r = random.uniform(-1, 1)
        population_i[i,:]  = population[i,:] + r*(elite_ind - population[j,:])
        population_i[i,-1] = target_function(population_i[i,0:population_i.shape[1]-1])
        x_list.append(i)
    population = np.vstack([population, population_i])
    population = population[population[:, -1].argsort()]
    population = population[:population_i.shape[0],:] 
    return population

# Function: Parasitism Phase
def parasitism(population, eta = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population_i = np.copy(population)
    population_i = mutation(population_i, 1, eta, min_values, max_values, target_function)
    population   = np.vstack([population, population_i])
    population   = population[population[:, -1].argsort()]
    population   = population[:population_i.shape[0],:] 
    return population

############################################################################

# SOS Function
def symbiotic_organisms_search(population_size = 5, min_values = [-5,-5], max_values = [5,5], eta = 1, generations = 50, target_function = target_function, verbose = True):    
    count      = 0
    population = initial_population(population_size, min_values, max_values, target_function)
    elite_ind  = np.copy(population[population[:,-1].argsort()][0,:])
    while (count <= generations):  
        if (verbose == True):
            print('Generation = ', count, ' f(x) = ', elite_ind[-1])  
        population = mutualism(population, elite_ind, target_function)
        population = comensalism(population, elite_ind, target_function)
        population = parasitism(population, eta, min_values, max_values, target_function)
        value      = np.copy(population[population[:,-1].argsort()][0,:])
        if(elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value) 
        count = count + 1       
    return elite_ind 

############################################################################