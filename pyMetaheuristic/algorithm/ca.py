############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Cultural Algorithm

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function):
    population     = np.random.uniform(min_values, max_values, (size, len(min_values)))
    fitness_values = np.apply_along_axis(target_function, 1, population)
    population     = np.hstack((population, fitness_values[:, np.newaxis]))
    return population

############################################################################

# Function: Binary Tournament
def binary_tournament(population):
    idx1, idx2 = np.random.choice(population.shape[0], 2, replace = False)
    return population[idx1] if population[idx1, -1] < population[idx2, -1] else population[idx2]

# Function: Update Belief Space Normative
def update_beliefspace_normative(min_values, max_values, accepted):
    for i in range(0, len(min_values)):
        min_values[i] = accepted[:, i].min()
        max_values[i] = accepted[:, i].max()
    return min_values, max_values

############################################################################

# Function: CA
def cultural_algorithm(size = 200, num_acc_ratio = 0.20, iterations = 50, min_values = [-100, -100], max_values = [100, 100], target_function = target_function, verbose = True):
    num_accepted = int(size * num_acc_ratio)
    population   = initial_variables(size, min_values, max_values, target_function)
    best         = population[np.argmin(population[:, -1])]
    count        = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration: ', count, ' f(x) = ', best[-1])
        children               = initial_variables(size, min_values, max_values, target_function)
        best_child             = children[np.argmin(children[:, -1])]
        best                   = best_child if best_child[-1] < best[-1] else best
        combined               = np.vstack((population, children))
        population             = np.array([binary_tournament(combined) for _ in range(0, size)])
        accepted               = population[np.argsort(population[:, -1], axis = 0)[:num_accepted]]
        min_values, max_values = update_beliefspace_normative(min_values, max_values, accepted)
        count                  = count  + 1
    return best

############################################################################