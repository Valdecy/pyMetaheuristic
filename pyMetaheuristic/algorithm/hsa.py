############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Harmony Search Algorithm

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import random


############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((size, len(min_values)+1))
    for i in range(0, size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,:-1])
    return position

############################################################################

# Function: Create Harmony
def create_harmony(search_space, memories, consid_rate, adjust_rate, bw, min_values, max_values, target_function):
    new_harmony = np.zeros(len(search_space) + 1)
    for i in range(len(search_space)):
        if (random.random() < consid_rate):
            value = memories[random.randint(0, len(memories) - 1), i]
            if (random.random() < adjust_rate):
                value = value + bw * np.random.uniform(-1.0, 1.0)
            value = max(search_space[i][0], min(search_space[i][1], value))
        else:
            value =  np.random.uniform(search_space[i][0], search_space[i][1])
        new_harmony[i] = value
    new_harmony[-1] = target_function(np.clip(new_harmony[:-1], min_values, max_values))
    return new_harmony

############################################################################

# Function: Harmony Search Algorithm
def harmony_search_algorithm(size = 200, min_values = [-100, -100], max_values = [100, 100], iterations = 25000, consid_rate = 0.95, adjust_rate = 0.7, bw = 0.05, target_function = target_function, verbose = True):
    search_space = list(zip(min_values, max_values))
    memories     = initial_variables(size, min_values, max_values, target_function)
    fond_memory  = memories[memories[:, -1].argmin()]
    count      = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iterations: ', count, ' f(x) = ', fond_memory[-1])
        harmony = create_harmony(search_space, memories, consid_rate, adjust_rate, bw, min_values, max_values, target_function)
        if (harmony[-1] < fond_memory[-1]):
            fond_memory = harmony
        memories = np.vstack([memories, harmony])
        memories = memories[memories[:, -1].argsort()][:size]
        count    = count + 1 
    return fond_memory

############################################################################

