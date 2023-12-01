############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Memetic Algorithm

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

# Function: Fitness
def fitness_function(population): 
    fitness = np.zeros((population.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ population[i,-1] + abs(population[:,-1].min()))
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

# Function: Offspring
def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1, elite = 0, target_function = target_function):
    offspring   = np.copy(population)
    b_offspring = 0
    if (elite > 0):
        preserve = np.copy(population[population[:,-1].argsort()])
        for i in range(0, elite):
            for j in range(0, offspring.shape[1]):
                offspring[i,j] = preserve[i,j]
    for i in range (elite, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - 1):
            rand   = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)   
            rand_c = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)                              
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            if (rand_c >= 0.5):
                offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            else:   
                offspring[i,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])  
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1]) 
    return offspring

# Function: Crossover Hill Clibing
def xhc(offspring, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1, target_function = target_function):
    offspring_xhc = np.zeros((2, len(min_values) + 1))
    b_offspring   = 0
    for _ in range (0, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(offspring) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - 1):
            rand   = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring_xhc[0,j] = np.clip(((1 + b_offspring)*offspring[parent_1, j] + (1 - b_offspring)*offspring[parent_2, j])/2, min_values[j], max_values[j])           
            offspring_xhc[1,j] = np.clip(((1 - b_offspring)*offspring[parent_1, j] + (1 + b_offspring)*offspring[parent_2, j])/2, min_values[j], max_values[j])           
        offspring_xhc[0,-1] = target_function(offspring_xhc[0,0:offspring_xhc.shape[1]-1])
        offspring_xhc[1,-1] = target_function(offspring_xhc[1,0:offspring_xhc.shape[1]-1]) 
        if (offspring_xhc[1,-1] < offspring_xhc[0,-1]):
            for k in range(0, offspring.shape[1]): 
                offspring_xhc[0, k] = offspring_xhc[1,k]            
        if (offspring[parent_1, -1] < offspring[parent_2, -1]):
            if (offspring_xhc[0,-1] < offspring[parent_1, -1]):
                for k in range(0, offspring.shape[1]): 
                    offspring[parent_1, k] = offspring_xhc[0,k]
        elif(offspring[parent_2, -1] < offspring[parent_1, -1]):
            if (offspring_xhc[0,-1] < offspring[parent_2, -1]):
                for k in range(0, offspring.shape[1]): 
                    offspring[parent_2, k] = offspring_xhc[0,k]
    return offspring

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
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
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1])                        
    return offspring

############################################################################

# MA Function
def memetic_algorithm(population_size = 5, mutation_rate = 0.1, elite = 0, min_values = [-5,-5], max_values = [5,5], eta = 1, mu = 1, std = 0.1, generations = 50, target_function = target_function, verbose = True):    
    count      = 0
    population = initial_population(population_size, min_values, max_values, target_function)
    fitness    = fitness_function(population)    
    elite_ind  = np.copy(population[population[:,-1].argsort()][0,:])  
    while (count <= generations): 
        if (verbose == True):
            print('Generation = ', count, ' f(x) = ', round(elite_ind[-1],4))        
        offspring  = breeding(population, fitness, min_values, max_values, mu, elite, target_function) 
        population = mutation(offspring, mutation_rate, eta, min_values, max_values, target_function)
        population = xhc(population, fitness, min_values, max_values, mu, target_function)
        if ((population[:,0:population.shape[1]-1].std())/len(min_values) < std):
            if (verbose == True):
                print('Reinitializing Population')
            population = initial_population(population_size, min_values, max_values, target_function)
        fitness = fitness_function(population)
        if(elite_ind[-1] > population[population[:,-1].argsort()][0,:][-1]):
            elite_ind = np.copy(population[population[:,-1].argsort()][0,:])    
        count = count + 1         
    return elite_ind 

############################################################################