############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Clonal Selection Algorithm

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import random

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population = np.zeros((size, len(min_values)+1))
    for i in range(0, size):
        for j in range(0, len(min_values)):
            population[i,j] = random.uniform(min_values[j], max_values[j])
        population[i,-1] = target_function(population[i,:-1])
    return population

############################################################################

# Function: Mutation Function
def mutate_vector(vector, m_rate):
    for i in range(0, len(vector)):
        if (random.random() < m_rate):
            vector[i] = vector[i]  + np.random.normal(0, 1)
    return vector

# Function: Clone and Hypermutate Function
def clone_and_hypermutate(population, clone_factor, min_values, max_values, target_function):
    clones     = []
    num_clones = int(population.shape[0] * clone_factor)
    for antibody in population:
        m_rate = np.exp(-2.5 * antibody[-1])
        for _ in range(0, num_clones):
            clone      = np.copy(antibody)
            clone[:-1] = np.clip(mutate_vector(clone[:-1], m_rate), min_values, max_values)
            clone[ -1] = target_function(clone[:-1])
            clones.append(clone)
    clones = np.array(clones)
    return clones

# Function: Random Insertion Function
def random_insertion(population, num_rand, min_values, max_values, target_function):
    if (num_rand == 0):
        return population
    for _ in range(0, num_rand):
        new_antibody = np.clip(np.random.uniform(min_values, max_values, len(min_values)), min_values, max_values)
        new_cost     = target_function(new_antibody)
        population   = np.vstack([population, np.append(new_antibody, new_cost)])
    return population

############################################################################

# Function: CLONALG
def clonal_selection_algorithm(size = 200, clone_factor = 0.1, num_rand = 2, iterations = 1000, min_values = [-100, -100], max_values = [100, 100], target_function = target_function, verbose = True):
    population = initial_variables(size, min_values, max_values, target_function)
    best       = population[np.argmin(population[:, -1])]
    count      = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration: ', count, ' f(x) = ', best[-1])
        clones     = clone_and_hypermutate(population, clone_factor, min_values, max_values, target_function)
        population = np.vstack([population, clones])
        population = population[np.argsort(population[:, -1])][:size]
        population = random_insertion(population, num_rand, min_values, max_values, target_function)
        best_      = population[np.argmin(population[:, -1])]
        best       = best_ if best_[-1] < best[-1] else best
        count      = count + 1
    return best

############################################################################

