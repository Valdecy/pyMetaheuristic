############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic:

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

# Function: Update Flames
def update_flames(flames, position):
    population = np.vstack([flames, position])
    flames     = np.copy(population[population[:,-1].argsort()][:flames.shape[0],:])
    return flames

# Function: Update Position
def update_position(position, flames, flame_number, b_constant, a_linear_component, min_values, max_values, target_function):
    num_rows, num_cols                 = position.shape[0], position.shape[1] - 1
    min_values                         = np.array(min_values)
    max_values                         = np.array(max_values)
    rnd_1                              = np.random.rand(num_rows, num_cols)
    rnd_2                              = (a_linear_component - 1) * rnd_1 + 1
    flame_distances                    = np.abs(flames[:flame_number, :num_cols] - position[:flame_number, :num_cols])
    position[:flame_number, :num_cols] = np.clip(flame_distances * np.exp(b_constant * rnd_2[:flame_number]) * np.cos(rnd_2[:flame_number] * 2 * np.pi) + flames[:flame_number, :num_cols], min_values, max_values)
    if (flame_number < num_rows):
        flame_distances                    = np.abs(flames[flame_number, :num_cols] - position[flame_number:, :num_cols])
        position[flame_number:, :num_cols] = np.clip(flame_distances * np.exp(b_constant * rnd_2[flame_number:]) * np.cos(rnd_2[flame_number:] * 2 * np.pi) + flames[flame_number, :num_cols], min_values, max_values)
    position[:, -1] = np.apply_along_axis(target_function, 1, position[:, :num_cols])
    return position

############################################################################

# Function: MFA
def moth_flame_algorithm(swarm_size = 15, min_values = [-100,-100], max_values = [100,100], generations = 50, b_constant = 1, target_function = target_function, verbose = True, start_init = None, target_value = None):
    count     = 0
    position  = initial_variables(swarm_size, min_values, max_values, target_function, start_init)
    flames    = np.copy(position[position[:,-1].argsort()][:,:])
    best_moth = np.copy(flames[0,:])
    while (count <= generations):
        if (verbose == True):
            print('Generation: ', count, ' f(x) = ', best_moth[-1])
        flame_number       = round(position.shape[0] - count*((position.shape[0] - 1)/generations))
        a_linear_component = -1 + count*((-1)/generations)
        position           = update_position(position, flames, flame_number, b_constant, a_linear_component, min_values, max_values, target_function)
        flames             = update_flames(flames, position)
        if (best_moth[-1] > flames[0, -1]):
            best_moth = np.copy(flames[0,:])      
        if (target_value is not None):
            if (best_moth[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1
    return best_moth

############################################################################