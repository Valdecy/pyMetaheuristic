############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Clonal Selection Algorithm

# PEREIRA, V. (2023). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np

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

# Function: Mutation Function
def mutate_vector(vector, m_rate):
    mutation_mask         = np.random.rand(len(vector)) < m_rate
    vector[mutation_mask] = vector[mutation_mask] + np.random.normal(0, 1, size=mutation_mask.sum())
    return vector

# Function: Clone and Hypermutate Function
def clone_and_hypermutate(population, clone_factor, min_values, max_values, target_function):
    num_clones     = int(population.shape[0] * clone_factor)
    mutation_rates = np.exp(-2.5 * population[:, -1])
    clones         = np.zeros((population.shape[0] * num_clones, population.shape[1]))
    for i, antibody in enumerate(population):
        start_idx                      = i * num_clones
        end_idx                        = start_idx + num_clones
        clones[start_idx:end_idx, :-1] = np.tile(antibody[:-1], (num_clones, 1))
        mutation_mask                  = np.random.rand(num_clones, len(min_values)) < mutation_rates[i]
        random_mutations               = np.random.normal(0, 1, (num_clones, len(min_values)))
        clones[start_idx:end_idx, :-1] = clones[start_idx:end_idx, :-1] + mutation_mask * random_mutations
        clones[start_idx:end_idx, :-1] = np.clip(clones[start_idx:end_idx, :-1], min_values, max_values)
    clones[:, -1] = np.apply_along_axis(target_function, 1, clones[:, :-1])
    return clones

# Function: Random Insertion Function
def random_insertion(population, num_rand, min_values, max_values, target_function):
    if (num_rand == 0):
        return population
    new_antibodies  = np.random.uniform(min_values, max_values, (num_rand, len(min_values)))
    new_antibodies  = np.clip(new_antibodies, min_values, max_values)
    new_costs       = np.apply_along_axis(target_function, 1, new_antibodies)
    new_individuals = np.hstack((new_antibodies, new_costs.reshape(-1, 1)))
    population      = np.vstack((population, new_individuals))
    return population

############################################################################

# Function: CLONALG
def clonal_selection_algorithm(size = 200, clone_factor = 0.1, num_rand = 2, iterations = 1000, min_values = [-100, -100], max_values = [100, 100], target_function = target_function, verbose = True, start_init = None, target_value = None):
    population = initial_variables(size, min_values, max_values, target_function, start_init)
    best       = population[np.argmin(population[:, -1])]
    count      = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration: ', count, ' f(x) = ', best[-1])
        clones     = clone_and_hypermutate(population, clone_factor, min_values, max_values, target_function)
        population = np.vstack([population, clones])
        population = population[np.argsort(population[:, -1])][:size]
        population = random_insertion(population, num_rand, min_values, max_values, target_function)
        best_      = population[np.argmin(population[:, -1])]
        best       = best_ if best_[-1] < best[-1] else best
        if (target_value is not None):
            if (best[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1  
    return best

############################################################################

