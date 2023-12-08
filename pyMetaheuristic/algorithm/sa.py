############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Simulated Annealing

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

# Function: Epson Vector
def epson_vector(min_values, mu, sigma):
    epson = np.random.normal(mu, sigma, len(min_values))
    return epson.reshape(1, -1)

# Function: Update Solution
def update_solution(guess, epson, min_values, max_values, target_function):
    updated_solution = guess[0,:-1] + epson
    updated_solution = np.clip(updated_solution, min_values, max_values)
    values           = np.apply_along_axis(target_function, 1, updated_solution)
    updated_solution = np.hstack((updated_solution, values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else values))
    return updated_solution

############################################################################

# Function: SA
def simulated_annealing(min_values = [-100,-100], max_values = [100,100], mu = 0, sigma = 1, initial_temperature = 1.0, temperature_iterations = 1000, final_temperature = 0.0001, alpha = 0.9, target_function = target_function, verbose = True, start_init = None, target_value = None):    
    guess       = initial_variables(1, min_values, max_values, target_function, start_init)
    best        = np.copy(guess)
    fx_best     = guess[0,-1]
    temperature = float(initial_temperature)
    while (temperature > final_temperature): 
        for repeat in range(0, temperature_iterations):
            if (verbose == True):
                print('Temperature = ', round(temperature, 4), ' ; iteration = ', repeat, ' ; f(x) = ', best[0, -1])
            fx_old    = guess[0,-1]    
            epson     = epson_vector(min_values, mu, sigma)
            new_guess = update_solution(guess, epson, min_values, max_values, target_function)
            fx_new    = new_guess[0,-1] 
            delta     = (fx_new - fx_old)
            r         = np.random.rand()
            p         = np.exp(-delta/temperature)
            if (delta < 0 or r <= p):
                guess = np.copy(new_guess)   
            if (fx_new < fx_best):
                fx_best = fx_new
                best    = np.copy(guess)
        temperature = alpha * temperature   
        if (target_value is not None):
            if (fx_best <= target_value):
                temperature = final_temperature 
                break
    return best[0, :]
    
############################################################################
