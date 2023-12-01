############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Chicken Swarm Optimization

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np
import random

from scipy.stats import norm

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

# Function: Update Positions
def update_chickens(population, min_values, max_values, roosters, hens, chicks, h_rooster, c_hen, best_chicken, target_function):
    e              = 1e-9
    old_population = np.copy(population)
    cut            = population.shape[0]
    for i in range(0, population.shape[0]):
        if (random.random() < 0.1):
            population[i,:-1] = np.random.uniform(min_values, max_values)
            population[i, -1] = target_function(population[i, :-1])
            continue
        for j in range(0, len(min_values)):
            if (population[i,:] in roosters):  # Rooster Behavior
                sigma_squared = 1
                delta         = best_chicken[-1] - population[i, -1]
                if (population[i, -1] < best_chicken[-1] and abs(delta / (abs(population[i, -1]) + e)) < 100) :
                    sigma_squared = np.exp(delta / (abs(population[i, -1]) + e))
                population[i, j] = np.clip(population[i, j] + norm.rvs(scale = sigma_squared), min_values[j], max_values[j])
            elif (population[i,:] in hens):  # Hen Behavior
                row = random.choice(population)
                try:
                    r1 = h_rooster[i]
                except:
                    k  = random.choice(list(h_rooster.keys()))
                    r1 = h_rooster[k]
                r2    = np.where(np.all(population == row, axis = 1))[0][0]
                delta = population[i, -1] - population[r1, -1]
                if (abs(delta / (abs(population[i, -1]) + e)) < 100):
                    s1 = np.exp(delta / (abs(population[i, -1]) + e))
                else:
                    s1 = 0
                population[i, j] = np.clip(population[i, j] + s1 * random.uniform(0, 1) * (population[r1, j] - population[i, j]) + random.uniform(0, 1) * (population[r2, j] - population[i, j]), min_values[j], max_values[j])
            elif (population[i,:] in chicks):  # Chick Behavior
                try:
                    m = c_hen[i]
                except: 
                    k  = random.choice(list(c_hen.keys()))
                    m = c_hen[k]
                population[i, j] = np.clip(population[i, j] + random.uniform(0.5, 0.9) * (population[m, j] - population[i, j]), min_values[j], max_values[j])
        population[i, -1] = target_function(population[i, :-1])
    population = np.vstack([np.unique(old_population, axis = 0), np.unique(population, axis = 0)])
    population = population[population[:, -1].argsort()]
    population = population[:cut, :]
    idx        = np.argmin(population[:, -1])
    if (population[idx, -1] < best_chicken[-1]):
        best_chicken = np.copy(population[idx, :])
    return best_chicken, population

############################################################################

# Function: Chicken SO
def chicken_swarm_optimization(size = 10, min_values = [-100, -100], max_values = [100, 100], g = 5, rooster_ratio = 0.2, hen_ratio = 0.6, generations = 150, target_function = target_function, verbose = True):
    population   = initial_variables(size, min_values, max_values, target_function) 
    idx          = np.argmin(population[:, -1])
    best_chicken = population[idx, :]
    count        = 0
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_chicken[-1])
        if (count % g == 0):
            population = population[population[:, -1].argsort()]
            roosters   = np.copy(population[:int(size * rooster_ratio), :])
            hens       = np.copy(population[ int(size * rooster_ratio):int(size*(rooster_ratio + hen_ratio)),:])
            chicks     = np.copy(population[ int(size * (rooster_ratio + hen_ratio)):, :])
            h_rooster  = {h: random.choice(range(int(size * rooster_ratio))) for h in range(int(size * rooster_ratio), int(size * rooster_ratio) + int(size * hen_ratio))}
            c_hen      = {c: random.choice(range(int(size * rooster_ratio), int(size * rooster_ratio) + int(size * hen_ratio))) for c in range(int(size * rooster_ratio) + int(size * hen_ratio), size)}
        best_chicken, population = update_chickens(population, min_values, max_values, roosters, hens, chicks, h_rooster, c_hen, best_chicken, target_function)
        count                    = count + 1
    return best_chicken

############################################################################
