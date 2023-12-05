############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Symbiotic Organisms Search

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

# Function: Mutation
def mutation(population, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    d_mutation = 0            
    for i in range (0, population.shape[0]):
        for j in range(0, population.shape[1] - 1):
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
                population[i,j] = np.clip((population[i,j] + d_mutation), min_values[j], max_values[j])
        population[i,-1] = target_function(population[i,0:population.shape[1]-1])                        
    return population

############################################################################

# Function: Mutualism Phase
def mutualism(population, elite_ind, target_function = target_function):
    x_list       = [i for i in range(0, population.shape[0])]
    population_i = np.copy(population)
    population_j = np.copy(population)
    for i in range(0, population.shape[0]):
        x_list.remove(i)
        j  = np.random.choice(x_list)
        r1 = np.random.rand()
        r2 = np.random.rand()
        mv = (population[i,:] + population[j,:])/2
        b1 = np.random.choice([1,2])
        b2 = np.random.choice([1,2])
        population_i[i,:]  = population[i,:] + r1*(elite_ind - mv*b1)
        population_j[i,:]  = population[j,:] + r2*(elite_ind - mv*b2)
        population_i[i,-1] = target_function(population_i[i,0:population_i.shape[1]-1])
        population_j[i,-1] = target_function(population_j[i,0:population_j.shape[1]-1])
        x_list.append(i)
    population = np.vstack([population, population_i, population_j])
    population = population[population[:, -1].argsort()]
    population = population[:population_i.shape[0],:] 
    return population

# Function: Comensalism Phase
def comensalism(population, elite_ind, target_function = target_function):
    x_list       = [i for i in range(0, population.shape[0])]
    population_i = np.copy(population)
    for i in range(0, population.shape[0]):
        x_list.remove(i)
        j = np.random.choice(x_list)
        r = np.random.uniform(-1, 1)
        population_i[i,:]  = population[i,:] + r*(elite_ind - population[j,:])
        population_i[i,-1] = target_function(population_i[i,0:population_i.shape[1]-1])
        x_list.append(i)
    population = np.vstack([population, population_i])
    population = population[population[:, -1].argsort()]
    population = population[:population_i.shape[0],:] 
    return population

# Function: Parasitism Phase
def parasitism(population, eta = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    population_i = np.copy(population)
    population_i = mutation(population_i, 1, eta, min_values, max_values, target_function)
    population   = np.vstack([population, population_i])
    population   = population[population[:, -1].argsort()]
    population   = population[:population_i.shape[0],:] 
    return population

############################################################################

# Function: SOS
def symbiotic_organisms_search(population_size = 5, min_values = [-100,-100], max_values = [100,100], eta = 1, generations = 50, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    population = initial_variables(population_size, min_values, max_values, target_function, start_init)
    elite_ind  = np.copy(population[population[:,-1].argsort()][0,:])
    count      = 0
    while (count <= generations):  
        if (verbose == True):
            print('Generation = ', count, ' f(x) = ', elite_ind[-1])  
        population = mutualism(population, elite_ind, target_function)
        population = comensalism(population, elite_ind, target_function)
        population = parasitism(population, eta, min_values, max_values, target_function)
        value      = np.copy(population[population[:,-1].argsort()][0,:])
        if(elite_ind[-1] > value[-1]):
            elite_ind = np.copy(value) 
        if (target_value is not None):
            if (elite_ind[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1       
    return elite_ind 

############################################################################