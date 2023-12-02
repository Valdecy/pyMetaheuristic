############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Adaptive Chaotic Grey Wolf Optimizer

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy  as np

############################################################################

# Function
def target_function():
    return

############################################################################

# Function: Initialize Variables
def initial_variables(size, min_values, max_values, target_function =  target_function, lmbda = 0.5):
    def chaotic_map(x, r):
        if (x < r):
            return x / r
        else:
            return (1 - x) / (1 - r)
    population = np.zeros((size, len(min_values)))
    for i in range(0, size):
        for j in range(0, len(min_values)):
            x                = chaotic_map(np.random.uniform(0, 1), lmbda)
            population[i, j] = min_values[j] + x * (max_values[j] - min_values[j])
    population     = np.clip(population, min_values, max_values)
    fitness_values = np.apply_along_axis(target_function, 1, population)
    population     = np.hstack((population, fitness_values[:, np.newaxis]))
    return population

############################################################################

# Function: Initialize Alpha
def alpha_position(min_values, max_values, target_function):
    alpha = np.zeros((1, len(min_values) + 1))
    for j in range(0, len(min_values)):
        alpha[0,j] = 0.0
    alpha[0,-1] = target_function(np.clip(alpha[0,0:alpha.shape[1]-1], min_values, max_values))
    return alpha[0,:]

# Function: Initialize Beta
def beta_position(min_values, max_values, target_function):
    beta = np.zeros((1, len(min_values) + 1))
    for j in range(0, len(min_values)):
        beta[0,j] = 0.0
    beta[0,-1] = target_function(np.clip(beta[0,0:beta.shape[1]-1], min_values, max_values))
    return beta[0,:]

# Function: Initialize Delta
def delta_position(min_values, max_values, target_function):
    delta =  np.zeros((1, len(min_values) + 1))
    for j in range(0, len(min_values)):
        delta[0,j] = 0.0
    delta[0,-1] = target_function(np.clip(delta[0,0:delta.shape[1]-1], min_values, max_values))
    return delta[0,:]

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i,-1] < alpha[-1]):
            alpha = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[-1] and updated_position[i,-1] < beta[-1]):
            beta = np.copy(updated_position[i,:])
        if (updated_position[i,-1] > alpha[-1] and updated_position[i,-1] > beta[-1]  and updated_position[i,-1] < delta[-1]):
            delta = np.copy(updated_position[i,:])
    return alpha, beta, delta

# Function: Update Position
def update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function):
    updated_position = np.copy(position)
    alpha_position   = np.copy(position)
    beta_position    = np.copy(position)
    delta_position   = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range (0, len(min_values)):   
            r1_alpha            = np.random.rand()
            r2_alpha            = np.random.rand()
            a_alpha             = 2 * a_linear_component * r1_alpha - a_linear_component
            c_alpha             = 2 * r2_alpha      
            distance_alpha      = abs(c_alpha * alpha[j] - position[i,j]) 
            x1                  = alpha[j] - a_alpha*distance_alpha   
            r1_beta             = np.random.rand()
            r2_beta             = np.random.rand()
            a_beta              = 2 * a_linear_component * r1_beta - a_linear_component
            c_beta              = 2 * r2_beta            
            distance_beta       = abs(c_beta*beta[j] - position[i,j]) 
            x2                  = beta[j] - a_beta*distance_beta                          
            r1_delta            = np.random.rand()
            r2_delta            = np.random.rand()
            a_delta             = 2 * a_linear_component * r1_delta - a_linear_component
            c_delta             = 2 * r2_delta            
            distance_delta      = abs(c_delta * delta[j] - position[i,j]) 
            x3                  = delta[j] - a_delta*distance_delta                                 
            alpha_position[i,j] = np.clip(x1, min_values[j], max_values[j])  
            beta_position [i,j] = np.clip(x2, min_values[j], max_values[j])  
            delta_position[i,j] = np.clip(x3, min_values[j], max_values[j])
        alpha_position[i,-1]    = target_function(alpha_position[i,:-1])
        beta_position [i,-1]    = target_function(beta_position [i,:-1])
        delta_position[i,-1]    = target_function(delta_position[i,:-1])
        w_alpha                 = alpha[-1] / (beta [-1] + 1e16)
        w_beta                  = beta [-1] / (delta[-1] + 1e16)
        updated_position[i,:-1] = np.clip((alpha_position[i,:-1] * w_alpha + beta_position[i,:-1]* w_beta + delta_position[i,:-1]) / 3, min_values, max_values)
        updated_position[i, -1] = target_function(updated_position[i,:-1])
    updated_position = np.vstack([position, updated_position, alpha_position, beta_position, delta_position])
    updated_position = updated_position[updated_position[:, -1].argsort()]
    updated_position = updated_position[:position.shape[0], :]
    return updated_position

############################################################################

# Function: ACGWO
def adaptive_chaotic_grey_wolf_optimizer(size = 15, lmbda = 0.5, min_values = [-100, -100], max_values = [100, 100], iterations = 1500, target_function = target_function, verbose = True):    
    count    = 0
    alpha    = alpha_position(min_values, max_values, target_function = target_function)
    beta     = beta_position (min_values, max_values, target_function = target_function)
    delta    = delta_position(min_values, max_values, target_function = target_function)
    position = initial_variables(size, min_values, max_values, target_function = target_function, lmbda = lmbda)
    while (count <= iterations): 
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', alpha[-1])      
        a_linear_component = 2 - 2 * ( (np.exp(count/iterations) - 1) / (np.exp(1) - 1) )
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        position           = update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function = target_function)    
        count              = count + 1          
    return alpha

############################################################################
