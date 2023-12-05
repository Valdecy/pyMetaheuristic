############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Flow Direction Algorithm

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

# Function: Create Neighbors
def neighbors(position_a, elite, beta, w_component, min_values, max_values, target_function):
    b_list = []
    for _ in range(0, beta):
        position_b = np.zeros((position_a.shape[0], len(min_values) + 1))
        for i in range(position_a.shape[0]):
            ru                 = np.random.rand(len(min_values))
            rn                 = np.random.normal(0, 1, len(min_values))
            ix                 = np.random.choice(np.delete(np.arange(position_a.shape[0]), i))
            dt                 = np.linalg.norm(position_a[i, :-1] - elite[:-1])
            dl                 = (ru * position_a[ix,  :-1] - ru * position_a[i, :-1]) * dt * w_component
            position_b[i, :-1] = np.clip(position_a[i, :-1] + rn * dl, min_values, max_values)
        position_b[:, -1] = np.apply_along_axis(target_function, 1, position_b[:, :-1])
        b_list.append(position_b)
    position_b = np.concatenate(b_list, axis = 0)
    return position_b

# Function:  Elite
def elite_flow(elite, elite_a, elite_b):
    if (elite_a[-1] < elite_b[-1] and elite[-1] > elite_a[-1]):
        elite = np.copy(elite_a)
    elif (elite_b[-1] < elite_a[-1] and elite[-1] > elite_b[-1]):
        elite = np.copy(elite_b)
    return elite

# Function: Distance Calculations
def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y))**2))

# Function: Updtade Position
def update_position(position_a, position_b, elite, min_values, max_values, target_function):
    dim = len(min_values)
    for k in range(0, position_b.shape[0], position_a.shape[0]):
        candidate = np.copy(position_a)
        for i in range(0, position_a.shape[0]):
            for j in range(0, dim):
                ix = np.random.choice(np.delete(np.arange(position_a.shape[0]), i))
                rn = np.random.normal(0, 1)
                if (position_b[i+k, -1] < position_a[i, -1]):
                    distance        = np.linalg.norm(position_a[i, :-1] - position_b[i+k, :-1])
                    slope           = (position_a[i, -1] - position_b[i+k, -1]) / (distance if distance != 0 else 1)
                    velocity        = rn * slope
                    candidate[i, j] = np.clip(position_a[i, j] + velocity * (position_a[i, j] - position_b[i+k, j]) / (distance if distance != 0 else 1), min_values[j], max_values[j])
                elif (position_a[ix, -1] < position_a[i, -1]):
                    candidate[i, j] = np.clip(position_a[i, j] + rn * (position_a[ix, j] - position_a[i, j]), min_values[j], max_values[j])
                else:
                    candidate[i, j] = np.clip(position_a[i, j] + 2 * rn * (elite[j] - position_a[i, j]), min_values[j], max_values[j])
            candidate[i, -1] = target_function(candidate[i, :-1])
            if (candidate[i, -1] < position_a[i, -1]):
                position_a[i, :] = candidate[i, :]
    return position_a
    
############################################################################

# Function: FDA
def flow_direction_algorithm(size = 25, beta = 8, min_values = [-5,-5], max_values = [5,5], iterations = 500, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    count      = 1
    position_a = initial_variables(size, min_values, max_values, target_function, start_init)
    position_b = initial_variables(size*beta, min_values, max_values, target_function)
    elite_a    = np.copy(position_a[position_a[:,-1].argsort()][0,:])
    elite_b    = np.copy(elite_a)
    elite      = np.copy(elite_a)
    while (count <= iterations):   
        if (verbose == True):    
            print('Iteration = ', count,  ' f(x) = ', elite[-1])
        r1          = np.random.rand()
        r2          = np.random.rand()
        w_component = ( (1 - count/iterations)**(2*r1) )*(r2*count/iterations)*r2
        elite       = elite_flow(elite, elite_a, elite_b)
        position_b  = neighbors(position_a, elite, beta, w_component, min_values, max_values, target_function)
        elite_b     = np.copy(position_b[position_b[:,-1].argsort()][0,:])
        elite       = elite_flow(elite, elite_a, elite_b)
        position_a  = update_position(position_a, position_b, elite, min_values, max_values, target_function)
        elite_a     = np.copy(position_a[position_a[:,-1].argsort()][0,:])
        elite       = elite_flow(elite, elite_a, elite_b)
        if (target_value is not None):
            if (elite[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1        
    return elite

############################################################################
