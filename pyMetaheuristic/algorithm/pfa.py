############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Pathfinder Algorithm

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
    return  population

############################################################################

# Function: Build Distance Matrix
def build_distance_matrix(population):
   a = np.copy(population[:,:-1])
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Update Pathfinder
def update_pathfinder(pathfinder, population, count, iterations, min_values = [-5, -5], max_values = [5, 5], target_function = target_function):
    u2         = np.random.uniform(-1, 1, 1)[0]
    r3         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    A          = u2*np.exp( (-2*count)/iterations )
    local_best = np.copy(population[population[:,-1].argsort()][ 0,:])
    pf         = pathfinder + 2*r3*(pathfinder - local_best) + A
    for j in range(0, len(min_values)):
        pf[j]  = np.clip(pf[j], min_values[j], max_values[j])
    pf[-1] = target_function(pf[0:pf.shape[0]-1])
    if (pathfinder[-1] > pf[-1]):
        pathfinder = np.copy(pf)  
    return pathfinder

# Function: Update Position
def update_position(pathfinder, population, count, iterations, min_values = [-5, -5], max_values = [5, 5], target_function = target_function):
    dist  = build_distance_matrix(population)
    u1    = np.random.uniform(-1, 1, 1)[0]
    r1    = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    r2    = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    alpha = np.random.uniform(1, 2, 1)[0]
    beta  = np.random.uniform(1, 2, 1)[0]
    new_p = np.zeros((population.shape[0]**2, population.shape[1]))
    c     = 0
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):  
            if (sum(population[i,:] - population[j,:]) == 0):
                new_p[c,:] = initial_population(1, min_values, max_values, target_function)
            else:
                e          = (1 - (count/iterations))*u1*dist[i, j]
                new_p[c,:] = population[i,:] + alpha*r1*(population[j,:]-population[i,:]) + beta*r2*(pathfinder-population[i,:]) + e
                for k in range(0, len(min_values)):
                    new_p[c, k] = np.clip(new_p[c,k], min_values[k], max_values[k])
            new_p[c,-1] = target_function(new_p[c, 0:new_p.shape[1]-1])
            c = c + 1
    new_p = np.vstack([population, new_p])
    new_p = new_p[new_p[:, -1].argsort()]
    new_p = new_p[:population.shape[0],:] 
    return new_p

############################################################################

# Function: PFA
def pathfinder_algorithm(population_size = 5, min_values = [-5, -5], max_values = [5, 5], iterations = 100, target_function = target_function, verbose = True):
    count      = 0
    population = initial_population(population_size, min_values, max_values, target_function)
    pathfinder = np.copy(population[population[:,-1].argsort()][ 0,:])
    while (count <= iterations):  
        if (verbose == True):
           print('Iteration = ', count, ' f(x) = ', pathfinder[-1])  
        population = update_position(pathfinder, population, count, iterations, min_values, max_values, target_function)
        value      = np.copy(population[population[:,-1].argsort()][ 0,:])
        if (pathfinder[-1] > value[-1]):
            pathfinder = np.copy(value)  
        pathfinder = update_pathfinder(pathfinder, population, count, iterations, min_values, max_values, target_function)
        count      = count + 1
    return pathfinder

############################################################################
