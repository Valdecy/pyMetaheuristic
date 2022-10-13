############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Krill Herd Algorithm

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
        population[i,-1] = target_function(population[i, 0:population.shape[1]-1])
    return population

############################################################################

# Function: Build Distance Matrix
def build_distance_matrix(population):
   a = np.copy(population[:,:-1])
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Movement 1 
def motion_induced_process(population, population_n, best_ind, worst_ind, c, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    n_max = 0.01
    dtm   = build_distance_matrix(population)
    np.fill_diagonal(dtm, float('+inf'))
    for i in range(0, population.shape[0]):
        idx  = np.argpartition(dtm[i,:], int(population.shape[1]/5))
        k_ij = 0
        x_ij = 0
        a_iL = 0
        for j in range(0, idx.shape[0]):
            k_ij = k_ij + (population_n[i, -1] - population_n[idx[j], -1])/(-best_ind[-1] + worst_ind[-1] + 0.00000000000000001)
            x_ij = x_ij + (population_n[idx[j], :-1] - population_n[i, :-1])/ (dtm[idx[j], i] + 0.00000000000000001)
            a_iL = a_iL + k_ij*x_ij 
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        a_i  = a_iL + 2*(rand + c)*best_ind[-1]*best_ind[:-1]
        for k in range(0, len(min_values)):
            w                 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            population_n[i,k] = np.clip(a_i[k]*n_max + w*population_n[i, k], min_values[k], max_values[k]) 
        population_n[i,-1] = target_function(population_n[i, 0:population_n.shape[1]-1])
    return population_n

# Function: Movement 2
def foraging_motion(population, population_f, best_ind, c, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for i in range(0, population.shape[0]):
        x_if = np.copy(best_ind[:-1])
        for k in range(0, len(min_values)):
            x_if[k] = x_if[k]*(1/best_ind[-1])
        x_if = x_if/( (1/best_ind[-1]) + 0.00000000000000001)
        k_if = target_function(x_if)
        v_f  = 0.02
        c_f  = 2*(1 - c)
        b_i  = c_f*k_if*x_if + best_ind[-1]*best_ind[:-1]
        for j in range(0, len(min_values)):
            w                  = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            population_f[i, j] = np.clip(v_f*b_i[j] + w*population_f[i, j], min_values[j], max_values[j]) 
        population_f[i,-1] = target_function(population_f[i, 0:population_f.shape[1]-1])
    return population_f

# Function: Movement 3
def physical_difusion(population, c, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population_d = np.zeros((population.shape[0], population.shape[1]))
    for i in range(0, population.shape[0]):
        d_max = random.uniform(0.002, 0.010)
        for j in range(0, population.shape[1]-1):
             population_d[i,j] = np.clip(d_max*random.uniform(-1, 1)*(1 - c), min_values[j], max_values[j]) 
        population_d[i,-1] = target_function(population_d[i, 0:population_d.shape[1]-1])
    return population_d

# Function: Update Position
def update_position(population, population_n, population_f, population_d, c_t, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    delta_t = 0
    for i in range(0, len(min_values)):
        delta_t = delta_t + (max_values[i] - min_values[i])
    delta_t = c_t*delta_t
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[1]-1):
            population[i,j] = np.clip( population[i,j] + delta_t*(population_n[i,j] + population_f[i,j] + population_d[i,j]), min_values[j], max_values[j]) 
    for i in range(0, population.shape[0]):
        population[i,-1] = target_function(population[i, 0:population.shape[1]-1])
    return population

############################################################################

# Function: Fitness
def fitness_function(population): 
    fitness = np.zeros((population.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ population[i,-1] + abs(population[:,-1].min()))
    fit_sum = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
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
 
# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand   = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
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

# Function: Krill Herd Algorithm
def krill_herd_algorithm(population_size = 5, min_values = [-5,-5], max_values = [5,5], generations = 10, c_t = 1, target_function = target_function, verbose = False):
    count        = 0
    c_t          = np.clip(c_t, 0, 2)
    population   = initial_population(population_size, min_values, max_values, target_function)
    best_ind     = np.copy(population[population[:,-1].argsort()][ 0,:])
    worst_ind    = np.copy(population[population[:,-1].argsort()][-1,:])
    population_n = np.zeros((population.shape[0], population.shape[1]))
    population_f = np.copy(population)
    population_d = np.zeros((population.shape[0], population.shape[1]))
    while (count <= generations):  
        c = count/generations 
        if (verbose == True):
           print('Generation = ', count, ' f(x) = ', best_ind[-1])   
        population_n  = motion_induced_process(population, population_n, best_ind, worst_ind, c, min_values, max_values, target_function)
        population_n  = population_n[population_n[:,-1].argsort()]
        value         = np.copy(population_n[0,:])
        if(best_ind[-1] > value[-1]):
            best_ind  = np.copy(value) 
        value         = np.copy(population_n[-1,:])
        if(worst_ind[-1] < value[-1]):
            worst_ind = np.copy(value) 
        population_f  = foraging_motion(population, population_f, best_ind, c, min_values, max_values, target_function)
        population_f  = population_f[population_f[:,-1].argsort()]
        value         = np.copy(population_f[0,:])
        if(best_ind[-1] > value[-1]):
            best_ind  = np.copy(value) 
        value         = np.copy(population_f[-1,:])
        if(worst_ind[-1] < value[-1]):
            worst_ind = np.copy(value) 
        population_d  = physical_difusion(population, c, min_values, max_values, target_function)
        population_d  = population_d[population_d[:,-1].argsort()]
        value         = np.copy(population_d[0,:])
        if(best_ind[-1] > value[-1]):
            best_ind  = np.copy(value) 
        value         = np.copy(population_d[-1,:])
        if(worst_ind[-1] < value[-1]):
            worst_ind = np.copy(value) 
        population    = update_position(population, population_n, population_f, population_d, c_t, min_values, max_values, target_function)
        population    = population[population[:,-1].argsort()]
        value         = np.copy(population[0,:])
        if(best_ind[-1] > value[-1]):
            best_ind  = np.copy(value) 
        value         = np.copy(population[-1,:])
        if(worst_ind[-1] < value[-1]):
            worst_ind = np.copy(value) 
        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
        if (rand <= 0.2):
            fitness    = fitness_function(population)
            offspring  = breeding(population, fitness, min_values, max_values, 1, 1, target_function) 
            population = mutation(offspring, 0.05, 1, min_values, max_values, target_function)
            population = population[population[:,-1].argsort()]
            value      = np.copy(population[0,:])
            if(best_ind[-1] > value[-1]):
                best_ind = np.copy(value) 
            value        = np.copy(population[-1,:])
            if(worst_ind[-1] < value[-1]):
                worst_ind = np.copy(value) 
        count = count + 1  
    return best_ind

############################################################################
