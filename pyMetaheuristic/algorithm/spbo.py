############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Student Psychology Based Optimization

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

# Function: Update Best Student
def update_best_student(population, best_student, min_values, max_values, target_function):
    idx  = list(range(0, population.shape[0]))
    idx  = np.random.choice(idx)
    k    = [1, 2]
    k    = np.random.choice(k)
    n_b  = np.copy(best_student)
    for j in range(0, len(min_values)):
        r1     = np.random.rand()
        n_b[j] = np.clip(n_b[j] + ((-1)**k)*r1*(n_b[j] - population[idx, j]), min_values[j], max_values[j])
    n_b[-1] = target_function(n_b[:-1]) 
    if (n_b[-1] > best_student[-1]):
       n_b = np.copy(best_student)
    return n_b

# Function: Update Good Student Type A
def update_good_student_a(population, best_student, type_idx, min_values, max_values, target_function):
    n_p = np.copy(population)
    for i in type_idx:
        for j in range(0, len(min_values)):
            r1        = np.random.rand()
            n_p[i, j] = np.clip(population[i, j] + r1*(best_student[j] - population[i, j]), min_values[j], max_values[j])
        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1])  
    return n_p[type_idx,:]

# Function: Update Good Student Type B
def update_good_student_b(population, best_student, type_idx, min_values, max_values, target_function):
    n_p = np.copy(population)
    for i in type_idx:
        for j in range(0, len(min_values)):
            r1        = np.random.rand()
            r2        = np.random.rand()
            n_p[i, j] = np.clip(population[i, j] + r1*(best_student[j] - population[i, j]) + (r2*(population[i, j] - population[type_idx, j].mean())), min_values[j], max_values[j])
        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1]) 
    return n_p[type_idx,:]

# Function: Update Average Student 
def update_average_student(population, type_idx, min_values, max_values, target_function):
    n_p = np.copy(population)
    for i in type_idx:
        for j in range(0, len(min_values)):
            r1        = np.random.rand()
            n_p[i, j] = np.clip(population[i, j] + r1*(np.mean(population[type_idx, j]) - population[i, j]), min_values[j], max_values[j])
        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1]) 
    return n_p[type_idx,:]

# Function: Update Random Student 
def update_random_student(population, type_idx, min_values, max_values, target_function):
    n_p = np.copy(population)
    for i in type_idx:
        for j in range(0, len(min_values)):
            r1        = np.random.rand()
            n_p[i, j] = np.clip(np.min(population[:, j]) + r1*(np.max(population[:, j]) - np.min(population[:, j])), min_values[j], max_values[j])
        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1]) 
    return n_p[type_idx,:]

############################################################################

# Function: Segments
def segments(population):
    n          = len(population) - 1
    x          = []
    a, b, c, d = 0, 0, 0, 0
    for i in range(0, n):
        a = i
        for j in range(i + 1, n):
            b = j
            for k in range(j + 1, n):
                c = k
                for L in range(k + 1, n + (i > 0)):
                    d = L
                    x.append((a, b, c, d))    
    return x

# Function: Classify Class
def classify_class(population, x):
    p_lst       = list(range(0, len(population)))
    y           = x[np.random.randint(len(x))]
    i, j, k, L  = y
    ga          = p_lst[ :i]
    gb          = p_lst[i:j]
    av          = p_lst[j:L]
    rd          = p_lst[L: ]
    return ga, gb, av, rd

############################################################################

# Function: SBPO
def student_psychology_based_optimization(population_size = 50, min_values = [-100,-100], max_values = [100,100], generations = 5000, target_function = target_function, verbose = True, start_init = None, target_value = None):
    size          = max(5, population_size)
    population    = initial_variables(size, min_values, max_values, target_function, start_init)
    best_student  = np.copy(population[population[:,-1].argsort()][0,:])
    x             = segments(population)
    count         = 0
    while (count <= generations):
        if (verbose == True):
            print('Generation = ', count, ' f(x) = ', best_student[-1]) 
        ga, gb, av, rd = classify_class(population, x)
        best_student   = update_best_student(population, best_student, min_values, max_values, target_function)
        population_1   = update_good_student_a(population, best_student, ga, min_values, max_values, target_function)
        population_2   = update_good_student_b(population, best_student, gb, min_values, max_values, target_function)
        population_3   = update_average_student(population, av, min_values, max_values, target_function)
        population_4   = update_random_student(population, rd, min_values, max_values, target_function)
        population     = np.vstack([population, population_1, population_2, population_3, population_4])
        population     = population[population[:, -1].argsort()]
        value          = np.copy(population[0,:])
        population     = population[:size,:] 
        if (best_student[-1] > value[-1]):
            best_student = np.copy(value) 
        if (target_value is not None):
            if (best_student[-1] <= target_value):
                count = 2* generations
            else:
                count = count + 1
        else:
            count = count + 1  
    return best_student

############################################################################
