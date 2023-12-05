############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Artificial Fish Swarm Algorithm

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

# Function: Distance Matrix
def build_distance_matrix(position):
   a = position[:,:-1]
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

############################################################################

# Function: Behavior - Prey
def prey(position, visual, attempts, min_values, max_values, target_function):
    position_ = np.copy(position)
    dim       = len(min_values)
    for i in range(0, position.shape[0]):
        count = 0
        while count <= attempts:
            k            = np.random.choice([c for c in range(position.shape[0]) if c != i])
            rand         = np.random.uniform(low = -1, high = 1, size = dim)
            new_position = np.clip(position[k, :dim] + visual * rand, min_values, max_values)
            new_fitness  = target_function(new_position)
            if (new_fitness <= position[i, -1]):
                position_[i, :dim] = new_position
                position_[i,   -1] = new_fitness
                break
            count = count + 1
        if (position_[i, -1] < position[i, -1]):
            position[i, :] = position_[i, :]
    position = np.vstack([position, position_])
    position = position[position[:, -1].argsort()]
    position = position[:position_.shape[0], :]
    return position

# Function: Behavior - Swarm
def swarm(position, visual, min_values, max_values, target_function):
    distance_matrix = build_distance_matrix(position)
    position_       = np.copy(position)
    dim             = len(min_values)
    rand            = np.random.uniform(low = -1, high = 1, size = (position.shape[0], dim))
    for i in range(0, position.shape[0]):
        visible_agents = np.where((distance_matrix[i, :] <= visual) & (np.arange(distance_matrix.shape[1]) != i))[0]
        if (len(visible_agents) == 0):
            visible_agents = np.array([c for c in range(position.shape[0]) if c != i])
        k            = np.random.choice(visible_agents)
        new_position = np.clip(position[k, :dim] + visual * rand[i, :], min_values, max_values)
        new_fitness  = target_function(new_position)
        if (new_fitness <= position[i, -1]):
            position_[i, :dim] = new_position
            position_[i,   -1] = new_fitness
    position = np.vstack([position, position_])
    position = position[position[:, -1].argsort()]
    position = position[:position_.shape[0], :]
    return position

# Function: Behavior - Follow
def follow(position, best_position, visual, min_values, max_values, target_function):
    position_     = np.copy(position)
    dim           = len(min_values)
    rand          = np.random.uniform(low = -1, high = 1, size = (position.shape[0], dim))
    new_positions = np.clip(best_position[:dim] + visual * rand, min_values, max_values)
    for i in range(0, position.shape[0]):
        new_fitness = target_function(new_positions[i])
        if (new_fitness <= position[i, -1]):
            position_[i, :dim] = new_positions[i]
            position_[i,   -1] = new_fitness
    position = np.vstack([position, position_])
    position = position[position[:, -1].argsort()]
    position = position[:position_.shape[0], :]
    return position

# Function: Behavior - Move
def move(position, visual, min_values, max_values, target_function):
    position_     = np.copy(position)
    dim           = len(min_values)
    rand          = np.random.uniform(low = -1, high = 1, size = (position.shape[0], dim))
    new_positions = np.clip(position[0, :dim] + visual * rand, min_values, max_values)
    for i in range(0, position.shape[0]):
        new_fitness = target_function(new_positions[i])
        if (new_fitness <= position[i, -1]):
            position_[i, :dim] = new_positions[i]
            position_[i,   -1] = new_fitness
    position = np.vstack([position, position_])
    position = position[position[:, -1].argsort()]
    position = position[:position_.shape[0], :]
    return position

# Function: Behavior - Leap
def leap(position, visual, min_values, max_values, target_function):
    dim           = len(min_values)
    rand          = np.random.uniform(low = -1, high = 1, size = (position.shape[0], dim))
    new_positions = np.clip(position[0, :dim] + visual * rand, min_values, max_values)
    for i in range(0, position.shape[0]):
        position[i, :dim] = new_positions[i]
        position[i,   -1] = target_function(position[i, :dim])
    return position

############################################################################

# Function: AFSA
def artificial_fish_swarm_algorithm(school_size = 25, attempts = 100, visual = 0.3, step = 0.5, delta = 0.5, min_values = [-5,-5], max_values = [5, 5], iterations = 1500, target_function = target_function, verbose = True, start_init = None, target_value = None):
    position      = initial_variables(school_size, min_values, max_values, target_function, start_init)
    best_position = np.copy(position[np.argmin(position[:,-1]),:]) 
    count         = 0
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_position[-1])
        position = prey(position, visual, attempts, min_values, max_values, target_function)
        position = swarm(position, visual, min_values, max_values, target_function)
        position = follow(position, best_position, visual, min_values, max_values, target_function)
        position = move(position, visual, min_values, max_values, target_function)
        position = leap(position, visual, min_values, max_values, target_function)
        if ( abs((sum(position[:,-1])/position.shape[0]) - best_position[-1]) <= delta):
            position = leap(position, visual, min_values, max_values, target_function)
        if (np.amin(position[:,-1]) < best_position[-1]):
            best_position = np.copy(position[np.argmin(position[:,-1]),:])  
        if (target_value is not None):
            if (best_position[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1
    return best_position

############################################################################