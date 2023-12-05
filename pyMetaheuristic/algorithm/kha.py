############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Krill Herd Algorithm

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
def build_distance_matrix(population):
   a = np.copy(population[:,:-1])
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Movement 1 
def motion_induced_process(population, population_n, best_ind, worst_ind, c, min_values, max_values, target_function):
    n_max        = 0.01
    min_values   = np.array(min_values)
    max_values   = np.array(max_values)
    num_features = population.shape[1] - 1
    dtm          = build_distance_matrix(population)
    np.fill_diagonal(dtm, float('+inf'))
    for i in range(population.shape[0]):
        idx                 = np.argpartition(dtm[i, :], int(num_features / 5))[:int(num_features / 5)]
        k_ij                = np.sum((population_n[i, -1] - population_n[idx, -1]) / (-best_ind[-1] + worst_ind[-1] + 1e-16))
        x_ij                = np.sum((population_n[idx, :-1] - population_n[i, :-1]) / (dtm[i, idx][:, np.newaxis] + 1e-16), axis = 0)
        a_iL                = k_ij * x_ij
        rand                = np.random.rand()
        a_i                 = a_iL + 2 * (rand + c) * best_ind[-1] * best_ind[:-1]
        w_vector            = np.random.rand(num_features)
        population_n[i,:-1] = np.clip(a_i * n_max + w_vector * population_n[i, :-1], min_values, max_values)
        population_n[i, -1] = target_function(population_n[i, :-1])
    return population_n

# Function: Movement 2
def foraging_motion(population, population_f, best_ind, c, min_values, max_values, target_function):
    min_values   = np.array(min_values)
    max_values   = np.array(max_values)
    num_features = len(min_values)
    x_if         = best_ind[:-1] * (1 / (best_ind[-1] + 1e-16))
    x_if         = x_if / (1 / (best_ind[-1] + 1e-16))
    k_if         = target_function(x_if)
    v_f          = 0.02
    c_f          = 2 * (1 - c)
    b_i          = c_f * k_if * x_if + best_ind[-1] * best_ind[:-1]
    for i in range(population.shape[0]):
        w_vector            = np.random.rand(num_features)
        population_f[i,:-1] = np.clip(v_f * b_i + w_vector * population_f[i, :-1], min_values, max_values)
        population_f[i, -1] = target_function(population_f[i, :-1])
    return population_f

# Function: Movement 3
def physical_difusion(population, c, min_values, max_values, target_function):
    min_values          = np.array(min_values)
    max_values          = np.array(max_values)
    num_features        = population.shape[1] - 1
    d_max               = np.random.uniform(0.002, 0.010, size = (population.shape[0], 1))
    random_diffusion    = np.random.uniform(-1, 1, size = (population.shape[0], num_features))
    population_d        = np.zeros_like(population)
    population_d[:,:-1] = np.clip(d_max * random_diffusion * (1 - c), min_values, max_values)
    for i in range(population.shape[0]):
        population_d[i, -1] = target_function(population_d[i, :-1])
    return population_d

# Function: Update Position
def update_position(population, population_n, population_f, population_d, c_t, min_values, max_values, target_function):
    old_population    = np.copy(population)
    min_values        = np.array(min_values)
    max_values        = np.array(max_values)
    delta_t           = c_t * np.sum(max_values - min_values)
    population[:,:-1] = np.clip(population[:, :-1] + delta_t * (population_n[:, :-1] + population_f[:, :-1] + population_d[:, :-1]), min_values, max_values)
    population[:, -1] = np.apply_along_axis(target_function, 1, population[:, :-1])
    population        = np.vstack([old_population, population, population_n, population_f, population_d])
    population        = population[population[:, -1].argsort()]
    population        = population[:old_population.shape[0], :]
    return population

############################################################################

# Function: Fitness
def fitness_function(population): 
    fitness = np.zeros((population.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ population[i,-1] + abs(population[:,-1].min()))
    fit_sum = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix     = 0
    random = np.random.rand()
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: Offspring
def breeding(population, fitness, min_values, max_values, mu, elite, target_function):
    offspring   = np.copy(population)
    b_offspring = 0
    if (elite > 0):
        preserve = np.copy(population[population[:,-1].argsort()])
        for i in range(0, elite):
            for j in range(0, offspring.shape[1]):
                offspring[i,j] = preserve[i,j]
    for i in range (elite, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = np.random.choice(len(population) - 1)
        for j in range(0, offspring.shape[1] - 1):
            rand   = np.random.rand()
            rand_b = np.random.rand() 
            rand_c = np.random.rand()                              
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            if (rand_c >= 0.5):
                offspring[i,j] = np.clip(((1 + b_offspring)*population[parent_1, j] + (1 - b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])           
            else:   
                offspring[i,j] = np.clip(((1 - b_offspring)*population[parent_1, j] + (1 + b_offspring)*population[parent_2, j])/2, min_values[j], max_values[j])
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1]) 
    return offspring
 
# Function: Mutation
def mutation(offspring, mutation_rate, eta, min_values, max_values, target_function):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - 1):
            probability = np.random.rand()
            if (probability < mutation_rate):
                rand   = np.random.rand()
                rand_d = np.random.rand()                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring[i,j] = np.clip((offspring[i,j] + d_mutation), min_values[j], max_values[j])
        offspring[i,-1] = target_function(offspring[i,0:offspring.shape[1]-1])                        
    return offspring

############################################################################

# Function: KHA
def krill_herd_algorithm(population_size = 15, min_values = [-100,-100], max_values = [100,100], generations = 500, c_t = 1, target_function = target_function, verbose = True, start_init = None, target_value = None):
    c_t          = np.clip(c_t, 0, 2)
    population   = initial_variables(population_size, min_values, max_values, target_function, start_init)
    best_ind     = np.copy(population[np.argmin(population[:, -1]), :])
    worst_ind    = np.copy(population[np.argmax(population[:, -1]), :])
    population_n = np.zeros_like(population)
    population_f = np.copy(population)
    population_d = np.zeros_like(population)
    count        = 0
    while (count <= generations):  
        c = count/generations 
        if (verbose == True):
           print('Generation = ', count, ' f(x) = ', best_ind[-1])   
        population_n = motion_induced_process(population, population_n, best_ind, worst_ind, c, min_values, max_values, target_function)
        population_f = foraging_motion(population, population_f, best_ind, c, min_values, max_values, target_function)
        population_d = physical_difusion(population, c, min_values, max_values, target_function)
        population   = update_position(population, population_n, population_f, population_d, c_t, min_values, max_values, target_function)
        rand         = np.random.rand()
        if (rand <= 0.2):
            fitness   = fitness_function(population)
            offspring  = breeding(population, fitness, min_values, max_values, 1, 1, target_function) 
            population = mutation(offspring, 0.05, 1, min_values, max_values, target_function)
        best_loc  = population[np.argmin(population[:, -1]), :]
        worst_ind = np.copy(population[np.argmax(population[:, -1]), :])
        if (best_loc[-1] < best_ind[-1]):
            best_ind = np.copy(best_loc)
        if (target_value is not None):
            if (best_ind[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1  
    return best_ind

############################################################################
