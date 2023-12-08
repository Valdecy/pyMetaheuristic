############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Whale Optimization Algorithm

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

# Function: Update Leader by Fitness
def update_leader(position, leader):
    best_idx = np.argmin(position[:, -1])
    if (position[best_idx, -1] < leader[-1]):
        leader = np.copy(position[best_idx, :])
    return leader

# Function: Update Position
def update_position(position, leader, a_linear_component, b_linear_component, spiral_param, min_values, max_values, target_function):
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    for i in range(position.shape[0]):
        r1_leader = np.random.rand()
        r2_leader = np.random.rand()
        a_leader  = 2 * a_linear_component * r1_leader - a_linear_component
        c_leader  = 2 * r2_leader
        p         = np.random.rand()
        for j in range(len(min_values)):
            if (p < 0.5):
                if (abs(a_leader) >= 1):
                    rand_leader_index = np.random.randint(0, position.shape[0])
                    x_rand            = position[rand_leader_index, :]
                    distance_x_rand   = np.abs(c_leader * x_rand[j] - position[i, j])
                    position[i, j]    = np.clip(x_rand[j] - a_leader * distance_x_rand, min_values[j], max_values[j])
                else:
                    distance_leader   = np.abs(c_leader * leader[j] - position[i, j])
                    position[i, j]    = np.clip(leader[j] - a_leader * distance_leader, min_values[j], max_values[j])
            else:
                distance_leader = np.abs(leader[j] - position[i, j])
                rand            = np.random.rand()
                m_param         = (b_linear_component - 1) * rand + 1
                position[i, j]  = np.clip(distance_leader * np.exp(spiral_param * m_param) * np.cos(m_param * 2 * np.pi) + leader[j], min_values[j], max_values[j])
        position[i, -1] = target_function(position[i, 0:position.shape[1] - 1])
    return position

############################################################################

# Function: WOA 
def whale_optimization_algorithm(hunting_party = 25, spiral_param = 1,  min_values = [-100,-100], max_values = [100,100], iterations = 500, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position = initial_variables(hunting_party, min_values, max_values, target_function, start_init)
    leader   = np.copy(position[position[:,-1].argsort()][0,:])
    count    = 0
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', leader[-1])
        a_linear_component =  2 - count*( 2/iterations)
        b_linear_component = -1 + count*(-1/iterations)
        leader             = update_leader(position, leader)
        position           = update_position(position, leader, a_linear_component, b_linear_component,  spiral_param, min_values, max_values, target_function)
        if (target_value is not None):
            if (leader[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1 
    return leader

############################################################################
