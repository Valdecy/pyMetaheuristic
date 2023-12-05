############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Artificial Bee Colony Optimization

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

# Function: Fitness Value
def fitness_calc(function_values):
    fitness_value = np.where(function_values >= 0, 1.0 / (1.0 + function_values), 1.0 + np.abs(function_values))
    return fitness_value

# Function: Fitness
def fitness_function(sources, fitness_calc):
    fitness_values     = fitness_calc(sources[:, -1])
    cumulative_sum     = np.cumsum(fitness_values)
    normalized_cum_sum = cumulative_sum / cumulative_sum[-1]
    fitness            = np.column_stack((fitness_values, normalized_cum_sum))
    return fitness

# Function: Selection
def roulette_wheel(fitness):
    ix = np.searchsorted(fitness[:, 1], np.random.rand())
    return ix

############################################################################

# Function: Employed Bee
def employed_bee(sources, min_values, max_values, target_function):
    searching_in_sources = np.copy(sources)
    dim                  = len(min_values)
    trial                = np.zeros((sources.shape[0], 1))
    new_solution         = np.zeros((1, dim))
    phi_values           = np.random.uniform(-1, 1, size = sources.shape[0])
    j_values             = np.random.randint(dim, size = sources.shape[0])
    k_values             = np.array([np.random.choice([k for k in range(sources.shape[0]) if k != i]) for i in range(sources.shape[0])])
    for i in range(0, sources.shape[0]):
        phi                = phi_values[i]
        j                  = j_values[i]
        k                  = k_values[i]
        xij                = searching_in_sources[i, j]
        xkj                = searching_in_sources[k, j]
        vij                = xij + phi * (xij - xkj)
        new_solution[0, :] = searching_in_sources[i, :-1]
        new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
        new_function_value = target_function(new_solution[0, :dim])
        if (fitness_calc(new_function_value) > fitness_calc(searching_in_sources[i, -1])):
            searching_in_sources[i, j] = new_solution[0, j]
            searching_in_sources[i,-1] = new_function_value
        else:
            trial[i, 0] = trial[i, 0] + 1
    return searching_in_sources, trial

# Function: Oulooker
def outlooker_bee(searching_in_sources, fitness, trial, min_values, max_values, target_function):
    improving_sources = np.copy(searching_in_sources)
    dim               = len(min_values)
    trial_update      = np.copy(trial)
    new_solution      = np.zeros((1, dim))
    phi_values        = np.random.uniform(-1, 1, size = improving_sources.shape[0])
    j_values          = np.random.randint(dim, size = improving_sources.shape[0])
    for repeat in range(0, improving_sources.shape[0]):
        i                  = roulette_wheel(fitness)
        phi                = phi_values[repeat]
        j                  = j_values[repeat]
        k                  = np.random.choice([k for k in range(0, improving_sources.shape[0]) if k != i])
        xij                = improving_sources[i, j]
        xkj                = improving_sources[k, j]
        vij                = xij + phi * (xij - xkj)
        new_solution[0, :] = improving_sources[i, :-1]
        new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
        new_function_value = target_function(new_solution[0, :dim])
        if (fitness_calc(new_function_value) > fitness_calc(improving_sources[i, -1])):
            improving_sources[i, j] = new_solution[0, j]
            improving_sources[i,-1] = new_function_value
            trial_update[i, 0]      = 0
        else:
            trial_update[i, 0]      = trial_update[i, 0] +  1
    return improving_sources, trial_update

# Function: Scouter
def scouter_bee(improving_sources, trial_update, limit, target_function):
    sources_to_update = np.where(trial_update > limit)[0]
    for i in sources_to_update:
        improving_sources[i, :-1] = np.random.normal(0, 1, improving_sources.shape[1] - 1)
        function_value            = target_function(improving_sources[i, 0:improving_sources.shape[1] - 1])
        improving_sources[i, -1]  = function_value
    return improving_sources

############################################################################

# Function: ABCO
def artificial_bee_colony_optimization(food_sources = 20, iterations = 100, min_values = [-5, -5], max_values = [5, 5], employed_bees = 5, outlookers_bees = 5, limit = 10, target_function = target_function, verbose = True, start_init = None, target_value = None):
    sources  = initial_variables(food_sources, min_values, max_values, target_function, start_init)
    fitness  = fitness_function(sources, fitness_calc)
    best_bee = sources[np.argmin(sources[:, -1]), :]
    count    = 0
    while count <= iterations:
        if count > 0 and verbose:
            print('Iteration = ', count, ' f(x) = ', best_bee[-1])
        e_bee = employed_bee(sources, min_values, max_values, target_function)
        for _ in range(employed_bees - 1):
            e_bee = employed_bee(e_bee[0], min_values, max_values, target_function)
        fitness = fitness_function(e_bee[0], fitness_calc)
        o_bee   = outlooker_bee(e_bee[0], fitness, e_bee[1], min_values, max_values, target_function)
        for _ in range(outlookers_bees - 1):
            o_bee = outlooker_bee(o_bee[0], fitness, o_bee[1], min_values, max_values, target_function)
        current_best_value = np.min(o_bee[0][:, -1])
        if (best_bee[-1] > current_best_value):
            best_bee = np.copy(o_bee[0][np.argmin(o_bee[0][:, -1]), :])
        sources = scouter_bee(o_bee[0], o_bee[1], limit, target_function)
        fitness = fitness_function(sources, fitness_calc)
        if (target_value is not None):
            if (best_bee[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1
    return best_bee

############################################################################