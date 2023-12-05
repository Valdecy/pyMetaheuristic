############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Metaheuristic: Geometric Mean Optimizer

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

# Function: Generate Guide
def generate_guide(pb, count, iterations, epsilon, min_values, max_values):
    ave      = np.mean(pb[:, -1])
    stdev    = np.std(pb[:, -1])
    n_best   = int(np.round(pb.shape[0] - (pb.shape[0] - 2) * (count / iterations)))
    dual_fit = np.zeros(pb.shape[0])
    for i in range(0, pb.shape[0]):
        div_1       = (stdev * np.sqrt(np.e)) 
        div_2       = (pb[:, -1] - ave)
        div_3       = (1 - np.eye(pb.shape[0])[i])
        exp_1       = np.exp( np.clip(-4 / (div_1 * div_2 + 1e-16), -5, 5) )
        dual_fit[i] = np.prod(1 / (1 + exp_1 * (div_3) ))
    index        = np.argsort(dual_fit)[::-1]
    sum_dual_fit = np.sum(dual_fit[index[:n_best]])
    pp_guide     = np.zeros_like(pb[:, :-1])
    for i in range(0, pb.shape[0]):
        for k in index[:n_best]:
            if (k != i):
                dfi         = dual_fit[k] / (sum_dual_fit + epsilon)
                pp_guide[i] = pp_guide[i] + dfi * pb[k, :-1]
        pp_guide[i] = np.clip(pp_guide[i], min_values, max_values)
    return pp_guide

# Function: Improve Guide
def improve_guide(pp_guide, pp, pb, target_function):
    pg  = np.copy(pp_guide)
    cut = pg.shape[0]
    fit = np.apply_along_axis(target_function, 1, pg)
    pg  = np.hstack((pg, fit[:, np.newaxis]))
    pg  = np.vstack([pg, pp, pb])
    pg  = pg[pg[:, -1].argsort()]
    pg  = pg[:cut,:-1]
    return pg

# Function: Update Population
def update_population(pp, pb, pv, pp_guide, best, min_values, max_values, min_vel, max_vel, w, target_function):
    for i in range(0, pp.shape[0]):
        mutant    = pp_guide[i] + w * np.random.randn(len(min_values)) * (np.max(pp_guide, axis = 0) - np.min(pp_guide, axis = 0))
        pv[i]     = w * pv[i] + (1 + (2 * np.random.rand(len(min_values)) - 1) * w) * (np.clip(mutant, min_values, max_values) - pp[i, :-1])
        pv[i]     = np.clip(pv[i], min_vel, max_vel)
        pp[i,:-1] = np.clip(pp[i, :-1] + pv[i], min_values, max_values)
        pp[i, -1] = target_function(pp[i, :-1])
        if (np.isnan(pp[i, -1])):
            pp[i,:-1] = min_values
            pp[i, -1] = target_function(pp[i, :-1])
        if (pp[i, -1] < pb[i, -1]):
            pb[i, :] = np.copy(pp[i, :])
            if (pp[i, -1] < best[-1]):
                best = np.copy(pp[i, :])
    return pp, pb, pv, best

############################################################################

# Function: GMO
def geometric_mean_optimizer(size = 15, iterations = 150, min_values = [-100, -100], max_values = [100, 100], epsilon = 0.001, target_function = target_function, verbose = True, start_init = None, target_value = None):
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    max_vel    = 0.1 * (max_values - min_values)
    min_vel    = -max_vel
    pp         = initial_variables(size, min_values, max_values, target_function, start_init)
    pb         = np.copy(pp)
    pv         = np.random.uniform(min_vel, max_vel, (size, len(min_values)))
    best       = pp[np.argmin(pp[:, -1])]
    count      = 0
    while (count <= iterations):
        if (verbose == True):    
            print('Iteration: ', count, ' f(x) = ', best[-1])
        pp_guide         = generate_guide(pb, count, iterations, epsilon, min_values, max_values)
        pp_guide         = improve_guide(pp_guide, pp, pb, target_function)
        w                = 1 - (count * (1 / iterations))
        pp, pb, pv, best = update_population(pp, pb, pv, pp_guide, best, min_values, max_values, min_vel, max_vel, w, target_function)
        if (target_value is not None):
            if (best[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1  
    return best

############################################################################

