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
import os

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
def euclidean_distance(x, y):
    distance = 0
    for j in range(0, len(x)):
        distance = (x[j] - y[j])**2 + distance       
    return distance**(1/2) 

# Function: Levy Distribution
def levy_flight(beta = 1.5):
    beta    = beta  
    r1      = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    r2      = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    sig_num = gamma(1+beta)*np.sin((np.pi*beta)/2.0)
    sig_den = gamma((1+beta)/2)*beta*2**((beta-1)/2)
    sigma   = (sig_num/sig_den)**(1/beta)
    levy    = (0.01*r1*sigma)/(abs(r2)**(1/beta))
    return levy

# Function: SAC
def separation_alignment_cohesion(dragonflies, radius, dragon = 0):
    dimensions = 0
    neighbours = 0
    index_list = []
    separation = np.zeros((1, dragonflies.shape[1]-1))
    alignment  = np.zeros((1, dragonflies.shape[1]-1))
    cohesion   = np.zeros((1, dragonflies.shape[1]-1))
    i          = dragon
    for j in range(0, dragonflies.shape[0]):
        if (i != j):
            for k in range(0, dragonflies.shape[1]-1):
                x  = dragonflies[i,0:dragonflies.shape[1]-1]
                y  = dragonflies[j,0:dragonflies.shape[1]-1]
                nd = euclidean_distance(x, y)
                if (nd < radius[0,k]):
                    dimensions      = dimensions + 1
                    separation[0,k] = separation[0,k] - dragonflies[i,k] - dragonflies[i,k]
                    alignment [0,k] = alignment [0,k] + dragonflies[j,k]
                    cohesion  [0,k] = cohesion  [0,k] + dragonflies[j,k]
            if (dimensions == dragonflies.shape[1] - 1):
                neighbours = neighbours + 1
                index_list.append(j)
    if (neighbours > 0):
        alignment = alignment/neighbours
        cohesion  = cohesion/neighbours
        for m in range(0, len(index_list)): 
            for n in range(0, dragonflies.shape[1]-1):
                cohesion[0,n] = cohesion[0,n] - dragonflies[index_list[m],n]                
    return separation, alignment, cohesion, neighbours

# Function: Update Food
def update_food(dragonflies, radius, food_position, min_values = [-5,-5], max_values = [5,5], dragon = 0, target_function = target_function):
    dimensions = 0 
    i          = dragon       
    x          = food_position[0,:-1]
    y          = dragonflies[i,:-1]
    fd         = euclidean_distance(x, y)
    for k in range(0, dragonflies.shape[1]-1):
        if (fd <= radius[0,k]):
            dimensions = dimensions + 1
    if (dimensions == dragonflies.shape[1] - 1):
        for k in range(0, dragonflies.shape[1]-1):
            food_position[0,k] = np.clip(food_position[0,k] - dragonflies[i,k], min_values[k], max_values[k])
    else:
        food_position[0,k] = 0           
    food_position[0,-1] = target_function(food_position[0,:-1])
    return food_position, dimensions

# Function: Update Predator
def update_predator(dragonflies, radius, predator, min_values = [-5,-5], max_values = [5,5], dragon = 0, target_function = target_function):
    dimensions = 0
    i          = dragon 
    x          = predator[0,:-1]
    y          = dragonflies[i,0:dragonflies.shape[1]-1]
    pd         = euclidean_distance(x, y)
    for k in range(0, dragonflies.shape[1]-1):
        if (pd <= radius[0,k]):
            dimensions = dimensions + 1
    if (dimensions == dragonflies.shape[1]):
        for k in range(0, dragonflies.shape[1]-1):
            predator[0,k] = np.clip(predator[0,k] + dragonflies[i,k], min_values[k], max_values[k])  
    else:
        predator[0,k] = 0
    predator[0,-1] = target_function(predator[0,0:predator.shape[1]-1])
    return predator

# Function: Update Search Matrices
def update_da(adjustment_const, weight_inertia, delta_max, dragonflies, best_dragon, radius, food_position, predator, delta_flies, min_values, max_values, target_function):
    rand1             = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    rand2             = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    rand3             = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    rand4             = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    weight_separation = 2*rand1*adjustment_const # Seperation Weight
    weight_alignment  = 2*rand2*adjustment_const # Alignment Weight
    weight_cohesion   = 2*rand3*adjustment_const # Cohesion Weight
    weight_food       = 2*rand4                  # Food Attraction Weight
    weight_predator   = 1*adjustment_const       # Enemy distraction Weight     
    for i in range(0, dragonflies.shape[0]):
        separation, alignment, cohesion, neighbours = separation_alignment_cohesion(dragonflies, radius, i)
        food_position, dimensions                   = update_food(dragonflies, radius, food_position, min_values, max_values, i, target_function)
        predator                                    = update_predator(dragonflies, radius, predator, min_values, max_values, i, target_function)
        if (dimensions > 0):
            if (neighbours >= 1):
                for j in range(0, len(min_values)):
                    r1               = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    r2               = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    r3               = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    delta_flies[i,j] = np.clip(weight_inertia*delta_flies[i,j] + r1*alignment[0, j] + r2*cohesion[0, j] + r3*separation[0, j], -delta_max[0,j], delta_max[0,j])
                    dragonflies[i,j] = dragonflies[i,j] + delta_flies[i,j]
            elif(neighbours < 1):
                for k in (0, dragonflies.shape[1]-1):
                    dragonflies[i,k] = dragonflies[i,k] + levy_flight(beta = 1.5)
                    delta_flies[i,k] = 0
        elif(dimensions == 0):
            for m in range(0, len(min_values)):
                delta_flies[i,m] = np.clip((weight_separation*separation[0,m] + weight_alignment*alignment[0,m] + weight_cohesion*cohesion[0,m]  + weight_food*food_position[0,m] + weight_predator*predator[0,m]) + weight_inertia*delta_flies[i,m], -delta_max[0,m], delta_max[0,m])
                dragonflies[i,m] = np.clip(dragonflies[i,m] + delta_flies[i,m], min_values[m], max_values[m])     
        dragonflies[i,-1] = target_function(dragonflies[i,:-1])
    for i in range (0, dragonflies.shape[0]):   
        if (dragonflies[i,-1] < food_position[0,-1]):
            for j in range(0, dragonflies.shape[1]):
                food_position[0,j] = dragonflies[i,j]
        if (dragonflies[i,-1] > predator[0,-1]):
            for j in range(0, dragonflies.shape[1]):
                predator[0,j] = dragonflies[i,j]  
    if (food_position[food_position[:,-1].argsort()][0,:][-1] < best_dragon[-1]):
        best_dragon = np.copy(food_position[food_position[:,-1].argsort()][0,:]) 
    else:
        for j in range(0, food_position.shape[1]):
            food_position[0,j] = best_dragon[j]
    return dragonflies, food_position, predator, delta_flies, best_dragon

############################################################################
  
# DA Function
def dragonfly_algorithm(size = 3, min_values = [-5,-5], max_values = [5,5], generations = 50, target_function = target_function, verbose = True):
    radius    = np.zeros((1, len(min_values)))
    delta_max = np.zeros((1, len(min_values)))
    for j in range(0, len(min_values)):
        radius[0,j]    = (max_values[j] - min_values[j])/10
        delta_max[0,j] = (max_values[j] - min_values[j])/10
    dragonflies   = initial_variables(size, min_values, max_values, target_function)
    delta_flies   = initial_variables(size, min_values, max_values, target_function)   
    predator      = initial_variables(1,    min_values, max_values, target_function)
    food_position = initial_variables(1,    min_values, max_values, target_function)   
    count         = 0
    for i in range (0, dragonflies.shape[0]):   
        if (dragonflies[i,-1] < food_position[0,-1]):
            for j in range(0, dragonflies.shape[1]):
                food_position[0,j] = dragonflies[i,j]
        if (dragonflies[i,-1] > predator[0,-1]):
            for j in range(0, dragonflies.shape[1]):
                predator[0,j] = dragonflies[i,j]          
    best_dragon = np.copy(food_position[food_position[:,-1].argsort()][0,:])   
    while (count <= generations):
        if (verbose == True):    
            print('Generation: ', count, ' f(x) = ', best_dragon[-1])
        for j in range(0, len(min_values)):
            radius[0,j]    = (max_values[j] - min_values[j])/4 + ((max_values[j] - min_values[j])*(count/generations)*2)
        weight_inertia   = 0.9 - count*((0.5)/generations)        
        adjustment_const = 0.1 - count*((0.1)/(generations/2))
        if (adjustment_const < 0):
            adjustment_const = 0
        dragonflies, food_position, predator, delta_flies, best_dragon = update_da(adjustment_const, weight_inertia, delta_max, dragonflies, best_dragon, radius, food_position, predator, delta_flies, min_values, max_values, target_function)
        count = count + 1
    return best_dragon

############################################################################