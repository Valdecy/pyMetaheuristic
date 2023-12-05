############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com 
# Metaheuristic: Random Search

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

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

# Function: Steps
def step(position, min_values, max_values, step_size, target_function):
    position_ = np.copy(position)
    rand      = np.random.rand(position.shape[0], position.shape[1] - 1)
    for j in range(0, position.shape[1] - 1):
        minimun         = np.clip(position[:, j] - step_size[:, j], min_values[j], max_values[j])
        maximum         = np.clip(position[:, j] + step_size[:, j], min_values[j], max_values[j])
        position_[:, j] = np.clip(minimun + (maximum - minimun) * rand[:, j], min_values[j], max_values[j])
    position_[:, -1] = np.apply_along_axis(target_function, 1, position_[:, :-1])
    return position_

# Function: Large Steps
def large_step(position, min_values, max_values, step_size, count, large_step_threshold, factor_1, factor_2, target_function):
    dim       = len(min_values)
    position_ = np.copy(position)
    factor    = factor_1 if count > 0 and count % large_step_threshold == 0 else factor_2
    adj_step  = np.array(step_size) * factor
    rand      = np.random.rand(position.shape[0], dim)
    for j in range(dim):
        minimum         = np.clip(position[:, j] - adj_step[:, j], min_values[j], max_values[j])
        maximum         = np.clip(position[:, j] + adj_step[:, j], min_values[j], max_values[j])
        position_[:, j] = np.clip(minimum + (maximum - minimum) * rand[:, j], min_values[j], max_values[j])
    position_[:, -1] = np.apply_along_axis(target_function, 1, position_[:, :-1])
    return adj_step, position_

# Function: Update Position
def update_position(solutions, position, position_step, position_large_step, step_size, step_large, threshold, improvement_threshold, factor_2):
    for i in range(0, solutions):
        if (position_step[i, -1] < position[i, -1] or position_large_step[i, -1] < position[i, -1]):
            position [i, :] = position_large_step[i, :] if position_large_step[i, -1] < position_step[i, -1] else position_step[i, :]
            step_size[i, :] = step_large[i, :] if position_large_step[i, -1] < position_step[i, -1] else step_size[i, :]
            threshold[i]    = 0
        else:
            threshold[i] = threshold[i]+ 1
            if (threshold[i] >= improvement_threshold):
                threshold[i]     = 0
                step_size[i, :] = step_size[i, :] / factor_2
    return position, step_size, threshold

############################################################################

# Function: ARS
def adaptive_random_search(solutions = 25, min_values = [-100,-100], max_values = [100,100], step_size_factor = 0.05, factor_1 = 3, factor_2 = 1.5, iterations = 500, large_step_threshold = 10, improvement_threshold = 25, target_function = target_function, verbose = True, start_init = None, target_value = None):
    threshold     = np.zeros(solutions)
    position      = initial_variables(solutions, min_values, max_values, target_function, start_init)
    best_solution = position[np.argmin(position[:, -1]), :]
    step_size     = np.array([(np.array(max_values) - np.array(min_values)) * step_size_factor for _ in range(solutions)])
    count         = 0
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_solution[-1])
        position_step                   = step(position, min_values, max_values, step_size, target_function)
        step_large, position_large_step = large_step(position, min_values, max_values, step_size, count, large_step_threshold, factor_1, factor_2, target_function)
        position, step_size, threshold  = update_position(solutions, position, position_step, position_large_step, step_size, step_large, threshold, improvement_threshold, factor_2) 
        best_index                      = np.argmin(position[:, -1])
        if (best_solution[-1] > position[best_index, -1]):
            best_solution = position[best_index, :]
        if (target_value is not None):
            if (best_solution[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1  
    return best_solution

############################################################################