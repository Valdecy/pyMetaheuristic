import numpy as np

# Function: EASOM. Solution ->  f(x1, x2) = -1; (x1, x2) = (3.14, 3.14)
def easom(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)
    return func_value
    
# Function: Six Hump Camel Back. Solution ->  f(x1, x2) = -1.0316; (x1, x2) = (0.0898, -0.7126) or (-0.0898, 0.7126)
def six_hump_camel_back(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = 4*x1**2 - 2.1*x1**4 + (1/3)*x1**6 + x1*x2 - 4*x2**2 + 4*x2**4
    return func_value

# Function: De Jong 1. Solution ->  f(xi) = 0; xi = 0
def de_jong_1(variables_values = [0, 0]):
    func_value = 0
    for i in range(0, len(variables_values)):
        func_value = func_value + variables_values[i]**2
    return func_value

# Function: Rosenbrocks Valley (De Jong 2). Solution ->  f(xi) = 0; xi = 1
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x     = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + 100*((variables_values[i] - (last_x)**2)**2) + (1 - last_x)**2
        last_x     = variables_values[i]
    return func_value
    
# Function: Axis Parallel Hyper-Ellipsoid. Solution ->  f(xi) = 0; xi = 0
def axis_parallel_hyper_ellipsoid(variables_values = [0, 0]):
    func_value = 0
    for i in range(0, len(variables_values)):
        func_value = func_value + i*variables_values[i]**2
    return func_value
    
# Function: Rastrigin. Solution ->  f(xi) = 0; xi = 0
def rastrigin(variables_values = [0, 0]):
    func_value = 10*len(variables_values)
    for i in range(0, len(variables_values)):
        func_value = func_value + (variables_values[i]**2) -10*np.cos(2*np.pi*variables_values[i])
    return func_value

# Function: Branin RCOS. Solution ->  f(xi) = 0.397887; (x1, x2) = (-3.14, 12.275) or (3.14, 2.275) or (9.42478, 2.475)
def branin_rcos(variables_values = [0, 0]):
    x1, x2     = variables_values
    a          = 1
    b          = 5.1/(4*np.pi**2)
    c          = 5/np.pi
    d          = 6
    e          = 10
    f          = 1/(8*np.pi)
    func_value = a*(x2 - b*x1**2 + c*x1 - d)**2 + e*(1 - f)*np.cos(x1) + e
    return func_value
    
# Function: Goldstein-Price. Solution ->  f(x1, x2) = 3; (x1, x2) = (0, -1)
def goldstein_price(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = (1 + ((x1 + x2 +1)**2)*(19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))*(30 + ((2*x1 - 3*x2)**2)*(18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return func_value
    
# Function: Styblinski Tang. Solution ->  f(xi) = -39.16599*number_of_variables; xi = -2.903534
def styblinski_tang(variables_values = [0, 0]):
    func_value = 0
    for i in range(0, len(variables_values)):
        func_value = func_value + variables_values[i]**4 - 16*variables_values[i]**2 + 5*variables_values[i]
    func_value = func_value*0.5    
    return func_value
