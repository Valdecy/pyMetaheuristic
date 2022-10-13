############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Artificial Fish Swarm Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

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
def initial_position(school_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((school_size, len(min_values)+1))
    for i in range(0, school_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# Function: Distance Matrix
def build_distance_matrix(position):
   a = position[:,:-1]
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

############################################################################

# Function: Behavior - Prey
def prey(position, visual, attempts, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position_ = np.copy(position)
    for i in range(0, position.shape[0]):
        count = 0
        while (count <= attempts):
            k = [c for c in range(0, position.shape[0]) if i != c]
            k = random.choice(k)
            for j in range(0, len(min_values)):
                rand            = np.random.uniform(low = -1, high = 1, size = 1)[0]
                position_[i, j] = np.clip(position[k, j] + visual*rand, min_values[j], max_values[j])
            position_[i,-1] = target_function(position_[i,0:position_.shape[1]-1])
            if (position_[i,-1] <= position[i,-1]):
                position[i,:] = position_[i,:]
                count         = attempts + 10
            count = count + 1
        if (count == attempts + 10):
            k = [c for c in range(0, position.shape[0]) if i != c]
            k = random.choice(k)
            for j in range(0, len(min_values)):
                rand            = np.random.uniform(low = -1, high = 1, size = 1)[0]
                position_[i, j] = np.clip(position[k, j] + visual*rand, min_values[j], max_values[j])
            position_[i,-1] = target_function(position_[i,0:position_.shape[1]-1])
            position[i,:]   = position_[i,:]
    return position

# Function: Behavior - Swarm
def swarm(position, visual, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    distance_matrix = build_distance_matrix(position)
    position_       = np.copy(position)
    for i in range(0, position.shape[0]):
        k = [c for c in range(0, distance_matrix.shape[1]) if distance_matrix[i, c] <= visual and i != c]
        if (len(k) == 0):
            k = [c for c in range(0, position.shape[0]) if i != c]
        k = random.choice(k)
        for j in range(0, len(min_values)):
            rand            = np.random.uniform(low = -1, high = 1, size = 1)[0]
            position_[i, j] = np.clip(position[k, j] + visual*rand, min_values[j], max_values[j])
        position_[i,-1] = target_function(position_[i,0:position_.shape[1]-1])
        if (position_[i,-1] <= position[i,-1]):
            position[i,:] = position_[i,:]
    return position

# Function: Behavior - Follow
def follow(position, best_position, visual, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position_  = np.copy(position)
    for i in range(0, position.shape[0]):
        k = [c for c in range(0, position.shape[0]) if i != c]
        k = random.choice(k)
        for j in range(0, len(min_values)):
            rand            = np.random.uniform(low = -1, high = 1, size = 1)[0]
            position_[i, j] = np.clip(best_position[0, j] + visual*rand, min_values[j], max_values[j])
        position_[i,-1] = target_function(position_[i,0:position_.shape[1]-1])
        if (position_[i,-1] <= position[i,-1]):
            position[i,:] = position_[i,:]
    return position

# Function: Behavior - Move
def move(position, visual, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position_  = np.copy(position)
    for i in range(0, position.shape[0]):
        k = [c for c in range(0, position.shape[0]) if i != c]
        k = random.choice(k)
        for j in range(0, len(min_values)):
            rand            = np.random.uniform(low = -1, high = 1, size = 1)[0]
            position_[i, j] = np.clip(position[0, j] + visual*rand, min_values[j], max_values[j])
        position_[i,-1] = target_function(position_[i,0:position_.shape[1]-1])
        if (position_[i,-1] <= position[i,-1]):
            position[i,:] = position_[i,:]
    return position

# Function: Behavior - Leap
def leap(position, visual, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for i in range(0, position.shape[0]):
        k = [c for c in range(0, position.shape[0]) if i != c]
        k = random.choice(k)
        for j in range(0, len(min_values)):
            rand           = np.random.uniform(low = -1, high = 1, size = 1)[0]
            position[i, j] = np.clip(position[0, j] + visual*rand, min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# AFSA Function
def artificial_fish_swarm_algorithm(school_size = 5, attempts = 100, visual = 0.3, step = 0.5, delta = 0.5, min_values = [-5,-5], max_values = [5,5], iterations = 1000, target_function = target_function, verbose = True):
    count         = 0
    position      = initial_position(school_size, min_values, max_values, target_function)
    best_position = np.copy(position[np.argmin(position[:,-1]),:].reshape(1,-1))   
    while (count <= iterations): 
        if (verbose == True):
            print('Iteration = ', count,  ' f(x) = ', best_position[0, -1])
        position = prey(position, visual, attempts, min_values, max_values, target_function)
        position = swarm(position, visual, min_values, max_values, target_function)
        position = follow(position, best_position, visual, min_values, max_values, target_function)
        position = move(position, visual, min_values, max_values, target_function)
        position = leap(position, visual, min_values, max_values, target_function)
        if ( abs((sum(position[:,-1])/position.shape[0]) - best_position[0,-1]) <= delta):
            position = leap(position, visual, min_values, max_values, target_function)
        if (np.amin(position[:,-1]) < best_position[0,-1]):
            best_position = np.copy(position[np.argmin(position[:,-1]),:].reshape(1,-1))  
        count    = count + 1
    return best_position

############################################################################