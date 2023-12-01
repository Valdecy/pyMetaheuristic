############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Dragonfly Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import random

from scipy.special import gamma

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

# Function: Distance Calculations
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2)) 

# Function: Levy Distribution
def levy_flight(d, beta, sigma):
    u    = np.random.randn(d) * sigma
    v    = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return 0.01 * step  

############################################################################

# Function: Update Food and Enemy Positions
def update_food_enemy(dragonflies, food_pos, enemy_pos):
    best_food_idx = np.argmin(dragonflies[:, -1])
    if (dragonflies[best_food_idx, -1] < food_pos[0, -1]):
        food_pos[0, :] = dragonflies[best_food_idx, :]
    worst_enemy_idx = np.argmax(dragonflies[:, -1])
    if (dragonflies[worst_enemy_idx, -1] > enemy_pos[0, -1]):
        enemy_pos[0, :] = dragonflies[worst_enemy_idx, :]
    return food_pos, enemy_pos

# Function: Update Search Matrices
def update_position(a, c, f, e, s, w, r, beta, sigma, enemy_pos, food_pos, delta_max, dragonflies, deltaflies, min_values, max_values, target_function):
    for i in range(dragonflies.shape[0]):
        neighbours_delta, neighbours_dragon = [], []
        for j in range(dragonflies.shape[0]):
            dist = euclidean_distance(dragonflies[i, :-1], dragonflies[j, :-1])
            if ( (dist > 0).all() & (dist <= r).all() ):
                neighbours_delta.append(deltaflies[j, :-1])
                neighbours_dragon.append(dragonflies[j, :-1])
        A      =  np.mean(neighbours_delta,  axis = 0) if neighbours_delta else deltaflies[i, :-1]
        C      =  np.mean(neighbours_dragon, axis = 0) - dragonflies[i, :-1] if neighbours_dragon else np.zeros(len(min_values))
        S      = -np.sum (neighbours_dragon - dragonflies[i, :-1], axis = 0) if neighbours_dragon else np.zeros(len(min_values))
        dist_f = euclidean_distance(dragonflies[i, :-1], food_pos [0, :-1])
        dist_e = euclidean_distance(dragonflies[i, :-1], enemy_pos[0, :-1])
        F      = food_pos [0, :-1] - dragonflies[i, :-1] if (dist_f <= r).all() else np.zeros(len(min_values))
        E      = enemy_pos[0, :-1] if (dist_e <= r).all() else np.zeros(len(min_values))
        for k in range(len(min_values)):
            if ( (dist_f > r).all() ):
                if len(neighbours_dragon) > 1:
                    deltaflies[i, k]    = w * deltaflies[i, k] + np.random.rand() * (a * A[k] + c * C[k] + s * S[k])
                else:
                    dragonflies[i, :-1] = dragonflies[i, :-1] + levy_flight(len(min_values), beta, sigma) * dragonflies[i, :-1]
                    deltaflies[i, k]    = np.clip(deltaflies[i, k], min_values[k], max_values[k])
                    break  
            else:
                deltaflies[i, k] = (a * A[k] + c * C[k] + s * S[k] + f * F[k] + e * E[k]) + w * deltaflies[i, k]
            deltaflies [i, k] = np.clip(deltaflies[i, k], -delta_max[k], delta_max[k])
            dragonflies[i, k] = dragonflies[i, k] + deltaflies[i, k]
            dragonflies[i, k] = np.clip(dragonflies[i, k], min_values[k], max_values[k])
        dragonflies[i, -1] = target_function(dragonflies[i, :-1])
    food_pos, enemy_pos = update_food_enemy(dragonflies, food_pos, enemy_pos)
    best_dragon         = np.copy(food_pos[food_pos[:, -1].argsort()][0, :])
    return enemy_pos, food_pos, dragonflies, deltaflies, best_dragon

############################################################################

# DA Function
def dragonfly_algorithm(size = 3, min_values = [-5,-5], max_values = [5,5], generations = 50, target_function = target_function, verbose = True):
    min_values  = np.array(min_values)
    max_values  = np.array(max_values) 
    delta_max   = (max_values - min_values) / 10
    food_pos    = initial_variables(1, min_values, max_values, target_function)
    enemy_pos   = initial_variables(1, min_values, max_values, target_function)
    dragonflies = initial_variables(size, min_values, max_values, target_function) 
    deltaflies  = initial_variables(size, min_values, max_values, target_function) 
    beta        = 3/2
    sigma       = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    count       = 0
    for i in range (0, dragonflies.shape[0]):   
        if (dragonflies[i,-1] > enemy_pos[0,-1]):
            for j in range(0, dragonflies.shape[1]):
                enemy_pos[0,j] = dragonflies[i,j]  
    best_dragon = np.copy(food_pos[food_pos[:,-1].argsort()][0,:])   
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_dragon[-1])
        r    = (max_values - min_values) / 4 + ((max_values - min_values) * count / generations * 2)
        w    = 0.9 - count * ((0.9 - 0.4) / generations)
        my_c = 0.1 - count * ((0.1 - 0  ) / (generations / 2))
        my_c = np.max(my_c, 0)
        s    = 2 * np.random.rand() * my_c  # Seperation Weight
        a    = 2 * np.random.rand() * my_c  # Alignment Weight
        c    = 2 * np.random.rand() * my_c  # Cohesion Weight
        f    = 2 * np.random.rand()         # Food Attraction Weight
        e    = my_c 
        food_pos, enemy_pos = update_food_enemy(dragonflies, food_pos, enemy_pos)      
        enemy_pos, food_pos, dragonflies, deltaflies, best_dragon = update_position(a, c, f, e, s, w, r, beta, sigma, enemy_pos, food_pos, delta_max, dragonflies, deltaflies, min_values, max_values, target_function)
        count = count + 1
    return best_dragon

############################################################################