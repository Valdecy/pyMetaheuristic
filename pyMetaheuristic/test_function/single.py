############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Test Functions

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic

############################################################################

# Required Libraries
import numpy as np

############################################################################

# Available Test Functions:

    # Ackley
    # Axis Parallel Hyper-Ellipsoid
    # Beale
    # Bohachevsky F1
    # Bohachevsky F2
    # Bohachevsky F3
    # Booth
    # Branin RCOS 
    # Bukin F6 
    # Cross in Tray
    # De Jong F1
    # Drop Wave
    # Easom
    # Eggholder
    # Goldstein-Price 
    # Griewangk F8
    # Himmelblau
    # Hölder Table
    # Matyas
    # McCormick
    # Lévi F13
    # Rastrigin 
    # Rosenbrocks Valley (De Jong F2)
    # Schaffer F2
    # Schaffer F4
    # Schaffer F6
    # Schwefel
    # Six Hump Camel Back
    # Styblinski-Tang
    # Three Hump Camel Back
    # Zakharov

############################################################################

# Function: Ackley. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -5 <= x1, x2 <= 5
def ackley(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -20*np.exp(-0.2*np.sqrt(0.5*(x1**2 + x2**2))) - np.exp(0.5*(np.cos(2*np.pi*x1) +np.cos(2*np.pi*x2) )) + np.exp(1) + 20
    return func_value

# Function: Axis Parallel Hyper-Ellipsoid. Solution -> f(xi) = 0; xi = 0. Domain -> -5.12 <= xi <= 5.12
def axis_parallel_hyper_ellipsoid(variables_values = [0, 0]):
    func_value = 0
    for i in range(0, len(variables_values)):
        func_value = func_value + (i+1)*variables_values[i]**2
    return func_value
    
# Function: Beale. Solution -> f(x1, x2) = 0; (x1, x2) = (3, 0.5). Domain -> -4.5 <= x1, x2 <= 4.5
def beale(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*(x2**2))**2 + (2.625 - x1 + x1*(x2**3))**2
    return func_value

# Function: Bohachevsky F1. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -100 <= x1, x2 <= 100
def bohachevsky_1(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7
    return func_value
    
# Function: Bohachevsky F2. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -100 <= x1, x2 <= 100
def bohachevsky_2(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1)*np.cos(4*np.pi*x2) + 0.3
    return func_value

# Function: Bohachevsky F3. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -100 <= x1, x2 <= 100
def bohachevsky_3(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1 + 4*np.pi*x2) + 0.3
    return func_value

# Function: Booth. Solution -> f(x1, x2) = 0; (x1, x2) = (1, 3). Domain -> -10 <= x1, x2 <= 10
def booth(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
    return func_value
    
# Function: Branin RCOS. Solution -> f(x1, x2) = 0.397887; (x1, x2) = (-3.14, 12.275) or (3.14, 2.275) or (9.42478, 2.475). Domain -> -5 <= x1 <= 10; 0 <= x2 <= 15
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
    
# Function: Bukin F6. Solution -> f(x1, x2) = 0; (x1, x2) = (-10, 1). Domain -> -15 <= x1 <= -5; -3 <= x2 <= 3
def bukin_6(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = 100*np.sqrt(abs(x2 - 0.01*(x1**2))) + 0.01*abs(x1 + 10)
    return func_value

# Function: Cross in Tray. Solution -> f(x1, x2) = -2.06261; (x1, x2) = (1.34941, 1.34941) or (-1.34941, 1.34941) or (1.34941, -1.34941) or (-1.34941, -1.34941). Domain -> -10 <= x1, x2 <= 10
def cross_in_tray(variables_values = [0, 0]):
    x1, x2     = variables_values
    a          = np.sin(x1)*np.sin(x2)
    b          = np.exp(abs(100 - np.sqrt(x1**2 + x2**2)/np.pi))
    func_value = -0.0001*(abs(a*b)+1)**0.1
    return func_value
    
# Function: De Jong F1. Solution -> f(xi) = 0; xi = 0. Domain -> -5.12 <= xi <= 5.12
def de_jong_1(variables_values = [0, 0]):
    func_value = 0
    for i in range(0, len(variables_values)):
        func_value = func_value + variables_values[i]**2
    return func_value

# Function: Drop Wave. Solution -> f(x1, x2) = -1; (x1, x2) = (0, 0). Domain -> -5.12 <= xi <= 5.12
def drop_wave(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = - (1 + np.cos(12*np.sqrt(x1**2 + x2**2))) / (0.5*(x1**2 + x2**2) + 2)
    return func_value
   
# Function: EASOM. Solution -> f(x1, x2) = -1; (x1, x2) = (3.14, 3.14). Domain -> -100 <= x1, x2 <= 100
def easom(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -np.cos(x1)*np.cos(x2)*np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)
    return func_value
    
# Function: Eggholder. Solution -> f(x1, x2) = -959.6407; (x1, x2) = (512, 404.2319). Domain -> -512 <= x1, x2 <= 512
def eggholder(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = - (x2 + 47)*np.sin(np.sqrt(abs( (x1/2) + x2 + 47))) - x1*np.sin(np.sqrt(abs( x1 - (x2 + 47))))
    return func_value
 
# Function: Goldstein-Price. Solution -> f(x1, x2) = 3; (x1, x2) = (0, -1). Domain -> -2 <= x1, x2 <= 2
def goldstein_price(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = (1 + ((x1 + x2 +1)**2)*(19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))*(30 + ((2*x1 - 3*x2)**2)*(18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return func_value

# Function: Griewangk F8. Solution -> f(xi) = 0; xi = 0. Domain -> -600 <= xi <= 600
def griewangk_8(variables_values = [0, 0]):
    fv_0 = 0
    fv_1 = 1
    for i in range(0, len(variables_values)):
        fv_0 = fv_0 + (variables_values[i]**2)/4000
        fv_1 = fv_1 * (np.cos(variables_values[i]/np.sqrt(i+1)))
    func_value = fv_0 - fv_1 + 1
    return func_value
  
# Function: Himmelblau. Solution -> f(x1, x2) = 0; (x1, x2) = (3, 2) or (-2.805118, 3.131312) or (-3.779310, -3.283186) or (3.584428 ,-1.848126). Domain -> -5 <= x1, x2 <= 5
def himmelblau(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    return func_value
    
# Function: Hölder Table. Solution -> f(x1, x2) = -19.2085; (x1, x2) = (8.05502, 9.66459) or (-8.05502, 9.66459) or (8.05502, -9.66459) or (-8.05502, -9.66459). Domain -> -10 <= x1, x2 <= 10
def holder_table(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -abs(np.sin(x1)*np.cos(x2)*np.exp(abs(1 - np.sqrt(x1**2 + x2**2)/np.pi)))
    return func_value
    
# Function: Matyas. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -10 <= x1, x2 <= 10
def matyas(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = 0.26*( x1**2 + x2**2 ) - 0.48*x1*x2
    return func_value

# Function: McCormick. Solution -> f(x1, x2) = -1.9133; (x1, x2) = (-0.54719, -1.54719). Domain -> -1.5 <= x1 <= 4; -3 <= x2 <= 4
def mccormick(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = np.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1
    return func_value
    
# Function: Lévi F13. Solution -> f(x1, x2) = 0; (x1, x2) = (1, 1). Domain -> -10 <= x1, x2 <= 10
def levi_13(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = (np.sin(3*np.pi*x1))**2 + ((x1 - 1)**2)*(1 + (np.sin(3*np.pi*x2)**2)) + ((x2 - 1)**2)*(1 +(np.sin(2*np.pi*x2)**2))
    return func_value
    
# Function: Rastrigin. Solution -> f(xi) = 0; xi = 0. Domain -> -5.12 <= xi <= 5.12
def rastrigin(variables_values = [0, 0]):
    func_value = 10*len(variables_values)
    for i in range(0, len(variables_values)):
        func_value = func_value + (variables_values[i]**2) -10*np.cos(2*np.pi*variables_values[i])
    return func_value

# Function: Rosenbrocks Valley (De Jong F2). Solution -> f(xi) = 0; xi = 1. Domain -> -inf<= xi <= +inf
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x     = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + 100*((variables_values[i] - (last_x)**2)**2) + (1 - last_x)**2
        last_x     = variables_values[i]
    return func_value

# Function: Schaffer F2. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -100 <= x1, x2 <= 100
def schaffer_2(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = 0.5 + ((np.sin(x1**2 - x2**2)**2) -0.5) / (1 + 0.001*(x1**2 + x2**2))**2
    return func_value
    
# Function: Schaffer F4. Solution -> f(x1, x2) = 0.292579; (x1, x2) = (0, 1.25313) or (0, -1.25313) or (1.25313, 0) or (-1.25313, 0). Domain -> -100 <= x1, x2 <= 100
def schaffer_4(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = 0.5 + (np.cos(np.sin(abs(x1**2 - x2**2)))**2 -0.5) / (1 + 0.001*(x1**2 + x2**2))**2
    return func_value
    
# Function: Schaffer F6. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -100 <= x1, x2 <= 100
def schaffer_6(variables_values = [0, 0]):
    x1, x2     = variables_values
    x          = (x1**2 + x2**2)
    func_value = 0.5 + ((np.sin(np.sqrt(x))**2) - 0.5) / (1 + 0.001 * x)**2
    return func_value

# Function: Schwefel. Solution -> f(x) = 0; xi = 420.9687. Domain -> -500 <= xi <= 500
def schwefel(variables_values = [0, 0]):
    fv_0       = 418.9829*len(variables_values)
    func_value = 0
    for i in range(0, len(variables_values)):
        func_value = func_value + variables_values[i]*np.sin(np.sqrt(abs(variables_values[i])))
    func_value = - func_value + fv_0
    return func_value
    
# Function: Six Hump Camel Back. Solution -> f(x1, x2) = -1.0316; (x1, x2) = (0.0898, -0.7126) or (-0.0898, 0.7126). Domain -> -3 <= x1 <= 3; -2 <= x2 <= 2
def six_hump_camel_back(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = 4*x1**2 - 2.1*x1**4 + (1/3)*x1**6 + x1*x2 - 4*x2**2 + 4*x2**4
    return func_value
    
# Function: Styblinski-Tang. Solution -> f(xi) = -39.16599*number_of_variables; xi = -2.903534. Domain -> -5 <= xi <= 5
def styblinski_tang(variables_values = [0, 0]):
    func_value = 0
    for i in range(0, len(variables_values)):
        func_value = func_value + variables_values[i]**4 - 16*variables_values[i]**2 + 5*variables_values[i]
    func_value = func_value*0.5    
    return func_value

# Function: Three Hump Camel Back. Solution -> f(x1, x2) = 0; (x1, x2) = (0, 0). Domain -> -5 <= x1, x2 <= 5
def three_hump_camel_back(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = 2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2
    return func_value

# Function: Zakharov. Solution -> f(xi) = 0; xi = 0. Domain -> -5 <= xi <= 10
def zakharov(variables_values = [0, 0]):
    fv_0 = 0
    fv_1 = 0
    fv_2 = 0
    for i in range(0, len(variables_values)):
        fv_0 = fv_0 + variables_values[i]**2
        fv_1 = fv_1 + 0.5*(i+1)*variables_values[i]
        fv_2 = fv_2 + 0.5*(i+1)*variables_values[i]
    func_value = fv_0 + fv_1**2 + fv_2**4
    return func_value
    
############################################################################
