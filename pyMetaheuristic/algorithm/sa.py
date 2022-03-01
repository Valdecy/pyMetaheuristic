############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Simulated Annealing

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Simulated_Annealing, File: Python-MH-Simulated Annealing.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Simulated_Annealing>

############################################################################

# Required Libraries
import math
import numpy  as np
import random
import os

# Function
def target_function():
    return

# Function: Initialize Variables
def initial_guess(min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    n = 1
    guess = np.zeros((n, len(min_values) + 1))
    for j in range(0, len(min_values)):
         guess[0,j] = random.uniform(min_values[j], max_values[j]) 
    guess[0,-1] = target_function(guess[0,0:guess.shape[1]-1])
    return guess

# Function: Epson Vector
def epson_vector(guess, mu = 0, sigma = 1):
    epson = np.zeros((1, guess.shape[1]-1))
    for j in range(0, guess.shape[1]-1):
        epson[0,j] = float(np.random.normal(mu, sigma, 1))
    return epson

# Function: Updtade Solution
def update_solution(guess, epson, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    updated_solution = np.copy(guess)
    for j in range(0, guess.shape[1] - 1):
        if (guess[0,j] + epson[0,j] > max_values[j]):
            updated_solution[0,j] = random.uniform(min_values[j], max_values[j])
        elif (guess[0,j] + epson[0,j] < min_values[j]):
            updated_solution[0,j] = random.uniform(min_values[j], max_values[j])
        else:
            updated_solution[0,j] = guess[0,j] + epson[0,j] 
    updated_solution[0,-1] = target_function(updated_solution[0,0:updated_solution.shape[1]-1])
    return updated_solution

# SA Function
def simulated_annealing(min_values = [-5,-5], max_values = [5,5], mu = 0, sigma = 1, initial_temperature = 1.0, temperature_iterations = 1000, final_temperature = 0.0001, alpha = 0.9, target_function = target_function):    
    guess = initial_guess(min_values = min_values, max_values = max_values, target_function = target_function)
    epson = epson_vector(guess, mu = mu, sigma = sigma)
    best  = np.copy(guess)
    fx_best = guess[0,-1]
    Temperature = float(initial_temperature)
    while (Temperature > final_temperature): 
        for repeat in range(0, temperature_iterations):
            print("Temperature = ", round(Temperature, 4), " ; iteration = ", repeat, " ; f(x) = ", round(best[0, -1], 4))
            fx_old    =  guess[0,-1]    
            epson     = epson_vector(guess, mu = mu, sigma = sigma)
            new_guess = update_solution(guess, epson, min_values = min_values, max_values = max_values, target_function = target_function)
            fx_new    = new_guess[0,-1] 
            delta     = (fx_new - fx_old)
            r         = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            p         = np.exp(-delta/Temperature)
            if (delta < 0 or r <= p):
                guess = np.copy(new_guess)   
            if (fx_new < fx_best):
                fx_best = fx_new
                best    = np.copy(guess)
        Temperature = alpha*Temperature
    print(best)    
    return best