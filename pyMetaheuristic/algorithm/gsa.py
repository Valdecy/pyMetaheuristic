############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Gravitational Search Algorithm

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np
import math
import random
import os

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_position(swarm_size = 3, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((swarm_size, len(min_values)+1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,:-1])
    return position

############################################################################

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

############################################################################

# Function: Force Calculation
def force_acting(position, mass_a, mass_p, g_const, k_best):
    eps  = 2.2204e-16
    r_ij = build_distance_matrix(position)
    f_i  = np.zeros((position.shape[0], position.shape[1]-1))
    for d in range(0, position.shape[1]-1):
        f_ij = np.zeros((position.shape[0], position.shape[0]))
        for i in range(0, f_ij.shape[0]):
            for j in range(0, f_ij.shape[1]):
                if (i != j):
                    f_ij[i,j] = g_const*( ( (mass_p[i]*mass_a[j])/(r_ij[i,j] + eps) ) * (position[j, d] - position[i, d]) ) 
                    f_ij[j,i] = f_ij[i,j]
        for i in range(0, f_i.shape[0]):
            for j in range(0, k_best):
                rand      = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                f_i[i, d] = f_i[i, d] + rand* f_ij[i,j]
    return f_i

# Function: Update Mass
def update_mass(position, best_t, worst_t):
    mass = np.zeros((position.shape[0], 1))
    for i in range(0, position.shape[0]):
        mass[i, 0] = (position[i,-1] - worst_t[-1])/(best_t[-1] - worst_t[-1])
    mass = mass/np.sum(mass)
    return mass

# Function: Updtade Velocity
def update_velocity(velocity, accelaration):
    new_velocity = np.copy(velocity)
    for i in range(0, velocity.shape[0]):
        for j in range(0, velocity.shape[1]):
            rand              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            new_velocity[i,j] = rand*velocity[i,j] + accelaration[i,j]
    return new_velocity

# Function: Updtade Position
def update_position(position, velocity, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    new_position = np.copy(position)
    for i in range(0, position.shape[0]):
        for j in range(0, position.shape[1] - 1):
            new_position[i,j] = np.clip((position[i,j] + velocity[i,j]),  min_values[j],  max_values[j])
        new_position[i,-1] = target_function(new_position[i,:-1])
        if (new_position[i,-1] < position[i,-1]):
            position[i,:] = np.copy(new_position[i,:])
    return position

############################################################################

# GSA Function
def gravitational_search_algorithm(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function, verbose = True):    
    count       = 0
    position    = initial_position(swarm_size, min_values, max_values, target_function)
    velocity    = np.zeros((position.shape[0], len(min_values)))
    best_global = np.copy(position[position[:,-1].argsort()][ 0,:])
    best_t      = np.copy(best_global)
    worst_t     = np.copy(position[position[:,-1].argsort()][-1,:])   
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_global[-1])
        g_const      = 100*math.exp(-20*(count/iterations))
        mass         = update_mass(position, best_t, worst_t)
        k_best       = position.shape[0] - (position.shape[0] * (count/iterations))
        force        = force_acting(position, mass, mass, g_const, np.clip(int(k_best), 1, position.shape[0]))
        accelaration = np.nan_to_num(force/mass)
        velocity     = update_velocity(velocity, accelaration)
        position     = update_position(position, velocity, min_values, max_values, target_function)   
        best_t       = np.copy(position[position[:,-1].argsort()][ 0,:])
        worst_t      = np.copy(position[position[:,-1].argsort()][-1,:])         
        if (best_global[-1] > best_t[-1]):
            best_global = np.copy(best_t)  
        count = count + 1       
    return best_global

############################################################################
