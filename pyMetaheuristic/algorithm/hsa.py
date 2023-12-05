############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Harmony Search Algorithm

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function, start_init = None):
    dim = len(min_values)
    if (start_init is not None):
        start_init = np.atleast_2d(start_init)
        n_rows     = size - start_init.shape[0]
        if (n_rows > 0):
            rows       = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = target_function(start_init) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, start_init)
        population     = np.hstack((start_init, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    else:
        population     = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = target_function(population) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, population)
        population     = np.hstack((population, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    return population

############################################################################

# Function: Create Harmony
def create_harmony(search_space, memories, consid_rate, adjust_rate, bw, min_values, max_values, target_function):
    new_harmony = np.zeros(len(search_space) + 1)
    for i in range(len(search_space)):
        if (np.random.rand() < consid_rate):
            value = memories[np.random.randint(0, len(memories) - 1), i]
            if (np.random.rand() < adjust_rate):
                value = value + bw * np.random.uniform(-1.0, 1.0)
            value = max(search_space[i][0], min(search_space[i][1], value))
        else:
            value =  np.random.uniform(search_space[i][0], search_space[i][1])
        new_harmony[i] = value
    new_harmony[-1] = target_function(np.clip(new_harmony[:-1], min_values, max_values))
    return new_harmony

############################################################################

# Function: HSA
def harmony_search_algorithm(size = 200, min_values = [-100, -100], max_values = [100, 100], iterations = 25000, consid_rate = 0.95, adjust_rate = 0.7, bw = 0.05, target_function = target_function, verbose = True, start_init = None, target_value = None):
    search_space = list(zip(min_values, max_values))
    memories     = initial_variables(size, min_values, max_values, target_function, start_init)
    fond_memory  = memories[memories[:, -1].argmin()]
    count        = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iterations: ', count, ' f(x) = ', fond_memory[-1])
        harmony = create_harmony(search_space, memories, consid_rate, adjust_rate, bw, min_values, max_values, target_function)
        if (harmony[-1] < fond_memory[-1]):
            fond_memory = harmony
        memories = np.vstack([memories, harmony])
        memories = memories[memories[:, -1].argsort()][:size]
        if (target_value is not None):
            if (fond_memory[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1     
    return fond_memory

############################################################################

