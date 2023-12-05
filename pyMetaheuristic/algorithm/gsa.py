############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Gravitational Search Algorithm

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
                rand      = np.random.rand()
                f_i[i, d] = f_i[i, d] + rand* f_ij[i,j]
    return f_i

# Function: Update Mass
def update_mass(position, best_t, worst_t):
    mass = (position[:, -1] - worst_t[-1]) / (best_t[-1] - worst_t[-1] + 1e-16)
    mass = mass / ( np.sum(mass) + 1e-16)
    return mass[:, np.newaxis]

# Function: Updtade Velocity
def update_velocity(velocity, acceleration):
    rand         = np.random.rand(*velocity.shape)
    new_velocity = rand * velocity + acceleration
    return new_velocity

# Function: Updtade Position
def update_position(position, velocity, min_values, max_values, target_function):
    old_position                = np.copy(position)
    new_position                = np.clip(position[:, :-1] + velocity, min_values, max_values)
    fitness_values              = np.apply_along_axis(target_function, 1, new_position)
    improved_mask               = fitness_values < position[:, -1]
    position[improved_mask,:-1] = new_position[improved_mask]
    position[improved_mask, -1] = fitness_values[improved_mask]
    position                    = np.vstack([position, old_position])
    position                    = position[np.argsort(position[:, -1])]
    position                    = position[:new_position.shape[0], :]
    return position

############################################################################

# Function: GSA
def gravitational_search_algorithm(swarm_size = 200, min_values = [-100,-100], max_values = [100,100], iterations = 1500, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    position    = initial_variables(swarm_size, min_values, max_values, target_function, start_init)
    velocity    = np.zeros((position.shape[0], len(min_values)))
    best_global = np.copy(position[position[:,-1].argsort()][ 0,:])
    best_t      = np.copy(best_global)
    worst_t     = np.copy(position[position[:,-1].argsort()][-1,:])  
    count       = 0
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_global[-1])
        g_const      = 100*np.exp(-20*(count/iterations))
        mass         = update_mass(position, best_t, worst_t)
        k_best       = position.shape[0] - (position.shape[0] * (count/iterations))
        force        = force_acting(position, mass, mass, g_const, np.clip(int(k_best), 1, position.shape[0]))
        accelaration = np.nan_to_num(force/(mass + 1e-16))
        velocity     = update_velocity(velocity, accelaration)
        position     = update_position(position, velocity, min_values, max_values, target_function)   
        best_t       = np.copy(position[position[:,-1].argsort()][ 0,:])
        worst_t      = np.copy(position[position[:,-1].argsort()][-1,:])         
        if (best_global[-1] > best_t[-1]):
            best_global = np.copy(best_t)  
        if (target_value is not None):
            if (best_global[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1       
    return best_global

############################################################################
