############################################################################
# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Test Functions

# PEREIRA, V. (2022). GitHub repository: https://github.com/Valdecy/pyMetaheuristic
############################################################################

# Required Libraries
import os
import numpy as np
from pathlib import Path

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
    # CEC Functions F1 - F12

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

def alpine_1(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(np.abs(np.sin(x) + 0.1*x)))


def alpine_2(variables_values = [1, 1]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.prod(np.sqrt(np.clip(x, 0.0, None))*np.sin(x)))


def bent_cigar(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(x[0]**2 + 1_000_000.0*np.sum(x[1:]**2))


def chung_reynolds(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(x**2)**2)


def cosine_mixture(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(-0.1*np.sum(np.cos(5*np.pi*x)) - np.sum(x**2))


def csendes(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    mask = x != 0
    return float(np.sum((x[mask]**6)*(2.0 + np.sin(1.0/x[mask]))))


def discus(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(1_000_000.0*x[0] + np.sum(x[1:]**2))


def dixon_price(variables_values = [1, 1]):
    x = np.asarray(variables_values, dtype=float)
    if x.size < 3:
        return float((x[0] - 1.0)**2)
    i = np.arange(2, x.size, dtype=float)
    return float((x[0] - 1.0)**2 + np.sum(i * (2.0*x[2:]**2 - x[1:x.size-1])**2))


def elliptic(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    if x.size == 1:
        return float(x[0]**2)
    i = np.arange(x.size, dtype=float)
    weights = 1_000_000.0**(i/(x.size - 1.0))
    return float(np.sum(weights*(x**2)))


def expanded_griewank_plus_rosenbrock(variables_values = [1, 1]):
    x = np.asarray(variables_values, dtype=float)
    if x.size < 2:
        return 0.0
    g = 100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0
    return float(np.sum((g**2)/4000.0 - np.cos(g/np.sqrt(np.arange(1, x.size))) ))


def happy_cat(variables_values = [0, 0], alpha = 0.25):
    x = np.asarray(variables_values, dtype=float)
    val1 = np.sum(np.abs(x*x - x.size)**alpha)
    val2 = np.sum((0.5*x*x + x)/x.size)
    return float(val1 + val2 + 0.5)


def hgbat(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    val1 = np.sum(x**2)
    val2 = np.sum(x)
    return float(np.sqrt(np.abs(val1**2 - val2**2)) + (0.5*val1 + val2)/x.size + 0.5)


def katsuura(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    k = np.arange(1, 33, dtype=float)[:, None]
    i = np.arange(1, x.size + 1, dtype=float)
    inner = np.round((2.0**k) * x) * (2.0 ** (-k))
    return float(np.prod(np.sum(inner, axis=0) * i + 1.0))


def levy(variables_values = [1, 1]):
    x = np.asarray(variables_values, dtype=float)
    w = 1.0 + (x - 1.0)/4.0
    term1 = np.sin(np.pi*w[0])**2
    wi = w[:-1]
    term2 = np.sum((wi - 1.0)**2 * (1.0 + 10.0*np.sin(np.pi*wi + 1.0)))
    term3 = (w[-1] - 1.0)**2 * (1.0 + np.sin(2.0*np.pi*w[-1])**2)
    return float(term1 + term2 + term3)


def michalewicz(variables_values = [2.20, 1.57], m = 10):
    x = np.asarray(variables_values, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(-np.sum(np.sin(x) * np.sin(i*(x**2)/np.pi)**(2.0*m)))


def perm(variables_values = None, beta = 0.5):
    if variables_values is None:
        variables_values = [1.0, 0.5]
    x = np.asarray(variables_values, dtype=float)
    ii = np.arange(1, x.size + 1, dtype=float)
    jj = np.tile(ii, (x.size, 1))
    x_matrix = np.tile(x, (x.size, 1))
    inner = np.sum((jj + beta) * (np.power(x_matrix, ii) - np.power(1.0 / jj, ii)), axis=0)
    return float(np.sum(inner**2))


def pinter(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    sub = np.roll(x, 1)
    add = np.roll(x, -1)
    i = np.arange(1, x.size + 1, dtype=float)
    a = sub*np.sin(x) + np.sin(add)
    b = sub**2 - 2.0*x + 3.0*add - np.cos(x) + 1.0
    return float(np.sum(i*x**2) + np.sum(20.0*i*np.sin(a)**2) + np.sum(i*np.log10(1.0 + i*b**2)))


def powell(variables_values = [0, 0, 0, 0]):
    x = np.asarray(variables_values, dtype=float)
    m = (x.size // 4) * 4
    if m == 0:
        return 0.0
    x = x[:m]
    x1, x2, x3, x4 = x[0::4], x[1::4], x[2::4], x[3::4]
    return float(np.sum((x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1 - x4)**4))


def qing(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    i = np.arange(1, x.size + 1, dtype=float)
    return float(np.sum((x**2 - i)**2))


def quintic(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(np.abs(x**5 - 3.0*x**4 + 4.0*x**3 + 2.0*x**2 - 10.0*x - 4.0)))


def ridge(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    inner = np.cumsum(x)
    return float(np.sum(inner**2))


def salomon(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    val = np.sqrt(np.sum(x**2))
    return float(1.0 - np.cos(2.0*np.pi*val) + 0.1*val)


def schumer_steiglitz(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(x**4))


def schwefel_221(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.max(np.abs(x)))


def schwefel_222(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(np.abs(x)) + np.prod(np.abs(x)))


def modified_schwefel(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    xx = x + 420.9687462275036
    conditions = [x > 500.0, x < -500.0]
    xx_mod = np.fmod(xx, 500.0)
    choices = [
        (500.0 - xx_mod) * np.sin(np.sqrt(np.abs(500.0 - xx_mod))) - (xx - 500.0) ** 2 / (10000.0 * x.size),
        (xx_mod - 500.0) * np.sin(np.sqrt(np.abs(xx_mod - 500.0))) + (xx - 500.0) ** 2 / (10000.0 * x.size),
    ]
    default = xx * np.sin(np.sqrt(np.abs(xx)))
    val = np.sum(np.select(conditions, choices, default=default))
    return float(418.9829 * x.size - val)


def sphere_2(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    i = np.arange(2, x.size + 2, dtype=float)
    return float(np.sum(np.abs(x)**i))


def sphere_3(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(np.cumsum(x**2)))


def step(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(np.floor(np.abs(x))))


def step_2(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(np.floor(x + 0.5)**2))


def step_3(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum(np.floor(x**2)))


def stepint(variables_values = [0, 0]):
    x = np.asarray(variables_values, dtype=float)
    return float(25.0 + np.sum(np.floor(x)))


def trid(variables_values = None):
    if variables_values is None:
        variables_values = [1.0, 1.0]
    x = np.asarray(variables_values, dtype=float)
    return float(np.sum((x - 1.0)**2) - np.sum(x[1:]*x[:-1]))


def weierstrass(variables_values = [0, 0], a = 0.5, b = 3.0, k_max = 20):
    x = np.asarray(variables_values, dtype=float)
    ks = np.arange(k_max + 1, dtype=float)
    t1 = sum((a**k)*np.cos(2.0*np.pi*(b**k)*(x + 0.5)) for k in ks)
    t2 = x.size * sum((a**k)*np.cos(np.pi*(b**k)) for k in ks)
    return float(np.sum(t1) - t2)


def whitley(variables_values = [1, 1]):
    x = np.asarray(variables_values, dtype=float)
    xi = np.tile(x, (x.size, 1)).T
    xj = np.tile(x, (x.size, 1))
    tmp = 100.0*(xi**2 - xj)**2 + (1.0 - xj)**2
    return float(np.sum((tmp**2)/4000.0 - np.cos(tmp) + 1.0))

############################################################################
# CEC 2022 Single Objective Bound Constrained Functions
# Origin: https://github.com/P-N-Suganthan/2022-SO-BO
#
# Notes:
# - These functions require the official CEC 2022 input_data files.
# - By default, this module looks for either:
#     1) a folder named "cec2022_input_data" next to this file, or
#     2) a folder named "input_data" next to this file, or
#     3) the directory pointed to by the environment variable
#        PYMETAHEURISTIC_CEC2022_DATA
# - Supported dimensions: D in {2, 10, 20}
# - Functions 6, 7, and 8 are only defined for D in {10, 20}
############################################################################

_CEC2022_DIMENSIONS = (2, 10, 20)
_CEC2022_NO_D2 = {6, 7, 8}


def _resolve_cec2022_data_dir():
    env_dir = os.environ.get("PYMETAHEURISTIC_CEC2022_DATA")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))
    here = Path(__file__).resolve().parent
    candidates.append(here / "cec2022_input_data")
    candidates.append(here / "input_data")
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "CEC 2022 data directory not found. Place the official input_data folder "
        "next to this file as 'cec2022_input_data' or set the environment variable "
        f"PYMETAHEURISTIC_CEC2022_DATA. Searched: {searched}"
    )


def _loadtxt_strict(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required CEC 2022 data file not found: {path}")
    return np.loadtxt(path)


def _shiftfunc_cec2022(x, nx, Os):
    x = np.asarray(x, dtype=float)
    Os = np.asarray(Os, dtype=float)
    return x[:nx] - Os[:nx]


def _rotatefunc_cec2022(x, nx, Mr):
    x = np.asarray(x, dtype=float)
    Mr = np.asarray(Mr, dtype=float)
    if Mr.ndim == 1:
        Mr = Mr.reshape((nx, nx))
    return Mr[:nx, :nx].dot(x[:nx])


def _sr_func_cec2022(x, nx, Os, Mr, sh_rate, s_flag, r_flag):
    x = np.asarray(x, dtype=float)
    if s_flag == 1:
        y = _shiftfunc_cec2022(x, nx, Os)
        y = y * sh_rate
    else:
        y = x[:nx] * sh_rate
    if r_flag == 1:
        return _rotatefunc_cec2022(y, nx, Mr)
    return y


def _ellips_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    return sum((10.0 ** (6.0 * i / (nx - 1))) * z[i] * z[i] for i in range(nx))


def _bent_cigar_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    return z[0] * z[0] + sum((10.0 ** 6.0) * z[i] * z[i] for i in range(1, nx))


def _discus_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    return (10.0 ** 6.0) * z[0] * z[0] + sum(z[i] * z[i] for i in range(1, nx))


def _rosenbrock_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 2.048 / 100.0, s_flag, r_flag)
    z = np.asarray(z, dtype=float).copy()
    z += 1.0
    f = 0.0
    for i in range(nx - 1):
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        f += 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    return f


def _ackley_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    sum1 = np.sum(z[:nx] ** 2)
    sum2 = np.sum(np.cos(2.0 * np.pi * z[:nx]))
    sum1 = -0.2 * np.sqrt(sum1 / nx)
    sum2 = sum2 / nx
    return np.e - 20.0 * np.exp(sum1) - np.exp(sum2) + 20.0


def _griewank_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 600.0 / 100.0, s_flag, r_flag)
    s = np.sum(z[:nx] ** 2)
    p = 1.0
    for i in range(nx):
        p *= np.cos(z[i] / np.sqrt(1.0 + i))
    return 1.0 + s / 4000.0 - p


def _rastrigin_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag)
    return float(np.sum((z[:nx] ** 2) - 10.0 * np.cos(2.0 * np.pi * z[:nx]) + 10.0))


def _schwefel_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1000.0 / 100.0, s_flag, r_flag)
    f = 0.0
    for i in range(nx):
        zi = z[i] + 4.209687462275036e+002
        if zi > 500:
            f -= (500.0 - np.fmod(zi, 500.0)) * np.sin(np.sqrt(500.0 - np.fmod(zi, 500.0)))
            tmp = (zi - 500.0) / 100.0
            f += tmp * tmp / nx
        elif zi < -500:
            f -= (-500.0 + np.fmod(np.fabs(zi), 500.0)) * np.sin(np.sqrt(500.0 - np.fmod(np.fabs(zi), 500.0)))
            tmp = (zi + 500.0) / 100.0
            f += tmp * tmp / nx
        else:
            f -= zi * np.sin(np.sqrt(np.fabs(zi)))
    f += 4.189828872724338e+002 * nx
    return f


def _grie_rosen_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
    z = np.asarray(z, dtype=float).copy()
    z += 1.0
    f = 0.0
    for i in range(nx - 1):
        tmp1 = z[i] * z[i] - z[i + 1]
        tmp2 = z[i] - 1.0
        temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
        f += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
    tmp1 = z[nx - 1] * z[nx - 1] - z[0]
    tmp2 = z[nx - 1] - 1.0
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    f += (temp * temp) / 4000.0 - np.cos(temp) + 1.0
    return f


def _escaffer6_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    f = 0.0
    for i in range(nx - 1):
        temp1 = np.sin(np.sqrt(z[i] * z[i] + z[i + 1] * z[i + 1])) ** 2
        temp2 = 1.0 + 0.001 * (z[i] * z[i] + z[i + 1] * z[i + 1])
        f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    temp1 = np.sin(np.sqrt(z[nx - 1] * z[nx - 1] + z[0] * z[0])) ** 2
    temp2 = 1.0 + 0.001 * (z[nx - 1] * z[nx - 1] + z[0] * z[0])
    f += 0.5 + (temp1 - 0.5) / (temp2 * temp2)
    return f


def _happycat_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    alpha = 1.0 / 8.0
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
    z = np.asarray(z, dtype=float).copy() - 1.0
    r2 = np.sum(z[:nx] ** 2)
    sum_z = np.sum(z[:nx])
    return np.abs(r2 - nx) ** (2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5


def _hgbat_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    alpha = 1.0 / 4.0
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
    z = np.asarray(z, dtype=float).copy() - 1.0
    r2 = np.sum(z[:nx] ** 2)
    sum_z = np.sum(z[:nx])
    return np.abs((r2 ** 2.0) - (sum_z ** 2.0)) ** (2 * alpha) + (0.5 * r2 + sum_z) / nx + 0.5


def _schaffer_F7_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    f = 0.0
    for i in range(nx - 1):
        zi = np.sqrt(z[i] * z[i] + z[i + 1] * z[i + 1])
        tmp = np.sin(50.0 * (zi ** 0.2))
        f += (zi ** 0.5) + (zi ** 0.5) * tmp * tmp
    return (f * f) / ((nx - 1) * (nx - 1))


def _step_rastrigin_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    y = np.asarray(x, dtype=float).copy()
    Os = np.asarray(Os, dtype=float)
    for i in range(nx):
        if np.fabs(y[i] - Os[i]) > 0.5:
            y[i] = Os[i] + np.floor(2.0 * (y[i] - Os[i]) + 0.5) / 2.0
    z = _sr_func_cec2022(y, nx, Os, Mr, 5.12 / 100.0, s_flag, r_flag)
    return float(np.sum((z[:nx] ** 2) - 10.0 * np.cos(2.0 * np.pi * z[:nx]) + 10.0))


def _levy_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    w = 1.0 + z[:nx] / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = ((w[nx - 1] - 1.0) ** 2) * (1.0 + (np.sin(2.0 * np.pi * w[nx - 1]) ** 2))
    sum_mid = 0.0
    for i in range(nx - 1):
        wi = w[i]
        sum_mid += ((wi - 1.0) ** 2) * (1.0 + 10.0 * (np.sin(np.pi * wi + 1.0) ** 2))
    return term1 + sum_mid + term3


def _zakharov_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    sum1 = np.sum(z[:nx] ** 2)
    sum2 = np.sum([0.5 * (i + 1) * z[i] for i in range(nx)])
    return sum1 + (sum2 ** 2) + (sum2 ** 4)


def _katsuura_func_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    z = _sr_func_cec2022(x, nx, Os, Mr, 5.0 / 100.0, s_flag, r_flag)
    f = 1.0
    tmp3 = (1.0 * nx) ** 1.2
    for i in range(nx):
        temp = 0.0
        for j in range(1, 33):
            tmp1 = 2.0 ** j
            tmp2 = tmp1 * z[i]
            temp += np.fabs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
        f *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)
    tmp1 = 10.0 / (nx * nx)
    return f * tmp1 - tmp1


def _hf02_cec2022(x, nx, Os, Mr, S, s_flag, r_flag):
    cf_num = 3
    Gp = [0.4, 0.4, 0.2]
    G_nx = [0] * cf_num
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G = [0] * cf_num
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    S = list(map(int, np.asarray(S).flatten()))
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[S[i] - 1]
    fit0 = _bent_cigar_func_cec2022(y[G[0]:G[1]], G_nx[0], Os, Mr, 0, 0)
    fit1 = _hgbat_func_cec2022(y[G[1]:G[2]], G_nx[1], Os, Mr, 0, 0)
    fit2 = _rastrigin_func_cec2022(y[G[2]:nx], G_nx[2], Os, Mr, 0, 0)
    return fit0 + fit1 + fit2


def _hf10_cec2022(x, nx, Os, Mr, S, s_flag, r_flag):
    cf_num = 6
    Gp = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]
    G_nx = [0] * cf_num
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G = [0] * cf_num
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    S = list(map(int, np.asarray(S).flatten()))
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[S[i] - 1]
    fits = [
        _hgbat_func_cec2022(y[G[0]:G[1]], G_nx[0], Os, Mr, 0, 0),
        _katsuura_func_cec2022(y[G[1]:G[2]], G_nx[1], Os, Mr, 0, 0),
        _ackley_func_cec2022(y[G[2]:G[3]], G_nx[2], Os, Mr, 0, 0),
        _rastrigin_func_cec2022(y[G[3]:G[4]], G_nx[3], Os, Mr, 0, 0),
        _schwefel_func_cec2022(y[G[4]:G[5]], G_nx[4], Os, Mr, 0, 0),
        _schaffer_F7_func_cec2022(y[G[5]:nx], G_nx[5], Os, Mr, 0, 0),
    ]
    return float(sum(fits))


def _hf06_cec2022(x, nx, Os, Mr, S, s_flag, r_flag):
    cf_num = 5
    Gp = [0.3, 0.2, 0.2, 0.1, 0.2]
    G_nx = [0] * cf_num
    tmp = 0
    for i in range(cf_num - 1):
        G_nx[i] = int(np.ceil(Gp[i] * nx))
        tmp += G_nx[i]
    G_nx[cf_num - 1] = nx - tmp
    G = [0] * cf_num
    for i in range(1, cf_num):
        G[i] = G[i - 1] + G_nx[i - 1]
    z = _sr_func_cec2022(x, nx, Os, Mr, 1.0, s_flag, r_flag)
    S = list(map(int, np.asarray(S).flatten()))
    y = np.zeros(nx)
    for i in range(nx):
        y[i] = z[S[i] - 1]
    fits = [
        _katsuura_func_cec2022(y[G[0]:G[1]], G_nx[0], Os, Mr, 0, 0),
        _happycat_func_cec2022(y[G[1]:G[2]], G_nx[1], Os, Mr, 0, 0),
        _grie_rosen_func_cec2022(y[G[2]:G[3]], G_nx[2], Os, Mr, 0, 0),
        _schwefel_func_cec2022(y[G[3]:G[4]], G_nx[3], Os, Mr, 0, 0),
        _ackley_func_cec2022(y[G[4]:nx], G_nx[4], Os, Mr, 0, 0),
    ]
    return float(sum(fits))


def _cf_cal_cec2022(x, nx, Os, delta, bias, fit, cf_num):
    INF = 1.0e99
    w = [0.0] * cf_num
    w_max = 0.0
    w_sum = 0.0
    for i in range(cf_num):
        fit[i] += bias[i]
        for j in range(nx):
            w[i] += (x[j] - Os[i * nx + j]) ** 2.0
        if w[i] != 0:
            w[i] = (1.0 / w[i]) ** 0.5 * np.exp(-w[i] / (2.0 * nx * (delta[i] ** 2.0)))
        else:
            w[i] = INF
        w_max = max(w_max, w[i])
    w_sum = sum(w)
    if w_max == 0:
        w = [1.0] * cf_num
        w_sum = float(cf_num)
    f = 0.0
    for i in range(cf_num):
        f += w[i] / w_sum * fit[i]
    return f


def _cf01_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    cf_num = 5
    fit = [0.0] * cf_num
    delta = [10, 20, 30, 40, 50]
    bias = [0, 200, 300, 100, 400]
    fit[0] = 10000 * _rosenbrock_func_cec2022(x, nx, Os[0 * nx:1 * nx], Mr[0 * nx:1 * nx, 0:nx], 1, r_flag) / 1e4
    fit[1] = 10000 * _ellips_func_cec2022(x, nx, Os[1 * nx:2 * nx], Mr[1 * nx:2 * nx, 0:nx], 1, r_flag) / 1e10
    fit[2] = 10000 * _bent_cigar_func_cec2022(x, nx, Os[2 * nx:3 * nx], Mr[2 * nx:3 * nx, 0:nx], 1, r_flag) / 1e30
    fit[3] = 10000 * _discus_func_cec2022(x, nx, Os[3 * nx:4 * nx], Mr[3 * nx:4 * nx, 0:nx], 1, r_flag) / 1e10
    fit[4] = 10000 * _ellips_func_cec2022(x, nx, Os[4 * nx:5 * nx], Mr[4 * nx:5 * nx, 0:nx], 1, 0) / 1e10
    return _cf_cal_cec2022(x, nx, Os, delta, bias, fit, cf_num)


def _cf02_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    cf_num = 3
    fit = [0.0] * cf_num
    delta = [20, 10, 10]
    bias = [0, 200, 100]
    fit[0] = _schwefel_func_cec2022(x, nx, Os[0 * nx:1 * nx], Mr[0 * nx:1 * nx, 0:nx], 1, 0)
    fit[1] = _rastrigin_func_cec2022(x, nx, Os[1 * nx:2 * nx], Mr[1 * nx:2 * nx, 0:nx], 1, r_flag)
    fit[2] = _hgbat_func_cec2022(x, nx, Os[2 * nx:3 * nx], Mr[2 * nx:3 * nx, 0:nx], 1, r_flag)
    return _cf_cal_cec2022(x, nx, Os, delta, bias, fit, cf_num)


def _cf06_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    cf_num = 5
    fit = [0.0] * cf_num
    delta = [20, 20, 30, 30, 20]
    bias = [0, 200, 300, 400, 200]
    fit[0] = 10000 * _escaffer6_func_cec2022(x, nx, Os[0 * nx:1 * nx], Mr[0 * nx:1 * nx, 0:nx], 1, r_flag) / 2e7
    fit[1] = _schwefel_func_cec2022(x, nx, Os[1 * nx:2 * nx], Mr[1 * nx:2 * nx, 0:nx], 1, r_flag)
    fit[2] = 1000 * _griewank_func_cec2022(x, nx, Os[2 * nx:3 * nx], Mr[2 * nx:3 * nx, 0:nx], 1, r_flag) / 100
    fit[3] = _rosenbrock_func_cec2022(x, nx, Os[3 * nx:4 * nx], Mr[3 * nx:4 * nx, 0:nx], 1, r_flag)
    fit[4] = 10000 * _rastrigin_func_cec2022(x, nx, Os[4 * nx:5 * nx], Mr[4 * nx:5 * nx, 0:nx], 1, r_flag) / 1e3
    return _cf_cal_cec2022(x, nx, Os, delta, bias, fit, cf_num)


def _cf07_cec2022(x, nx, Os, Mr, s_flag, r_flag):
    cf_num = 6
    fit = [0.0] * cf_num
    delta = [10, 20, 30, 40, 50, 60]
    bias = [0, 300, 500, 100, 400, 200]
    fit[0] = 10000 * _hgbat_func_cec2022(x, nx, Os[0 * nx:1 * nx], Mr[0 * nx:1 * nx, 0:nx], 1, r_flag) / 1000
    fit[1] = 10000 * _rastrigin_func_cec2022(x, nx, Os[1 * nx:2 * nx], Mr[1 * nx:2 * nx, 0:nx], 1, r_flag) / 1e3
    fit[2] = 10000 * _schwefel_func_cec2022(x, nx, Os[2 * nx:3 * nx], Mr[2 * nx:3 * nx, 0:nx], 1, r_flag) / 4e3
    fit[3] = 10000 * _bent_cigar_func_cec2022(x, nx, Os[3 * nx:4 * nx], Mr[3 * nx:4 * nx, 0:nx], 1, r_flag) / 1e30
    fit[4] = 10000 * _ellips_func_cec2022(x, nx, Os[4 * nx:5 * nx], Mr[4 * nx:5 * nx, 0:nx], 1, r_flag) / 1e10
    fit[5] = 10000 * _escaffer6_func_cec2022(x, nx, Os[5 * nx:6 * nx], Mr[5 * nx:6 * nx, 0:nx], 1, r_flag) / 2e7
    return _cf_cal_cec2022(x, nx, Os, delta, bias, fit, cf_num)


def _cec2022_eval(variables_values, func_num):
    x = np.asarray(variables_values, dtype=float).reshape(-1)
    nx = int(x.shape[0])

    if func_num < 1 or func_num > 12:
        raise ValueError(f"CEC 2022 function {func_num} is not defined. Valid IDs are 1..12.")
    if nx not in _CEC2022_DIMENSIONS:
        raise ValueError(f"CEC 2022 functions are only defined for D in {_CEC2022_DIMENSIONS}. Got D={nx}.")
    if nx == 2 and func_num in _CEC2022_NO_D2:
        raise ValueError("CEC 2022 functions 6, 7, and 8 are not defined for D=2.")

    data_dir = _resolve_cec2022_data_dir()
    M = _loadtxt_strict(data_dir / f"M_{func_num}_D{nx}.txt")
    OShift_temp = _loadtxt_strict(data_dir / f"shift_data_{func_num}.txt")

    if func_num < 9:
        OShift = np.asarray(OShift_temp).reshape(-1)[:nx]
    else:
        OShift = np.asarray(OShift_temp).reshape(-1)

    SS = None
    if func_num in (6, 7, 8):
        SS = _loadtxt_strict(data_dir / f"shuffle_data_{func_num}_D{nx}.txt")

    if func_num == 1:
        return float(_zakharov_func_cec2022(x, nx, OShift, M, 1, 1) + 300.0)
    if func_num == 2:
        return float(_rosenbrock_func_cec2022(x, nx, OShift, M, 1, 1) + 400.0)
    if func_num == 3:
        return float(_schaffer_F7_func_cec2022(x, nx, OShift, M, 1, 1) + 600.0)
    if func_num == 4:
        return float(_step_rastrigin_func_cec2022(x, nx, OShift, M, 1, 1) + 800.0)
    if func_num == 5:
        return float(_levy_func_cec2022(x, nx, OShift, M, 1, 1) + 900.0)
    if func_num == 6:
        return float(_hf02_cec2022(x, nx, OShift, M, SS, 1, 1) + 1800.0)
    if func_num == 7:
        return float(_hf10_cec2022(x, nx, OShift, M, SS, 1, 1) + 2000.0)
    if func_num == 8:
        return float(_hf06_cec2022(x, nx, OShift, M, SS, 1, 1) + 2200.0)
    if func_num == 9:
        return float(_cf01_cec2022(x, nx, OShift, M, 1, 1) + 2300.0)
    if func_num == 10:
        return float(_cf02_cec2022(x, nx, OShift, M, 1, 1) + 2400.0)
    if func_num == 11:
        return float(_cf06_cec2022(x, nx, OShift, M, 1, 1) + 2600.0)
    return float(_cf07_cec2022(x, nx, OShift, M, 1, 1) + 2700.0)


def cec_2022_f01(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 1)


def cec_2022_f02(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 2)


def cec_2022_f03(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 3)


def cec_2022_f04(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 4)


def cec_2022_f05(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 5)


def cec_2022_f06(variables_values=[0] * 10):
    return _cec2022_eval(variables_values, 6)


def cec_2022_f07(variables_values=[0] * 10):
    return _cec2022_eval(variables_values, 7)


def cec_2022_f08(variables_values=[0] * 10):
    return _cec2022_eval(variables_values, 8)


def cec_2022_f09(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 9)


def cec_2022_f10(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 10)


def cec_2022_f11(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 11)


def cec_2022_f12(variables_values=[0, 0]):
    return _cec2022_eval(variables_values, 12)


############################################################################

FUNCTIONS = {
    "ackley": ackley,
    "axis_parallel_hyper_ellipsoid": axis_parallel_hyper_ellipsoid,
    "beale": beale,
    "bohachevsky_1": bohachevsky_1,
    "bohachevsky_2": bohachevsky_2,
    "bohachevsky_3": bohachevsky_3,
    "booth": booth,
    "branin_rcos": branin_rcos,
    "bukin_6": bukin_6,
    "cross_in_tray": cross_in_tray,
    "de_jong_1": de_jong_1,
    "drop_wave": drop_wave,
    "easom": easom,
    "eggholder": eggholder,
    "goldstein_price": goldstein_price,
    "griewangk_8": griewangk_8,
    "himmelblau": himmelblau,
    "holder_table": holder_table,
    "matyas": matyas,
    "mccormick": mccormick,
    "levi_13": levi_13,
    "rastrigin": rastrigin,
    "rosenbrocks_valley": rosenbrocks_valley,
    "schaffer_2": schaffer_2,
    "schaffer_4": schaffer_4,
    "schaffer_6": schaffer_6,
    "schwefel": schwefel,
    "six_hump_camel_back": six_hump_camel_back,
    "styblinski_tang": styblinski_tang,
    "three_hump_camel_back": three_hump_camel_back,
    "zakharov": zakharov,

    "alpine_1": alpine_1,
    "alpine_2": alpine_2,
    "bent_cigar": bent_cigar,
    "chung_reynolds": chung_reynolds,
    "cosine_mixture": cosine_mixture,
    "csendes": csendes,
    "discus": discus,
    "dixon_price": dixon_price,
    "elliptic": elliptic,
    "expanded_griewank_plus_rosenbrock": expanded_griewank_plus_rosenbrock,
    "happy_cat": happy_cat,
    "hgbat": hgbat,
    "katsuura": katsuura,
    "levy": levy,
    "michalewicz": michalewicz,
    "perm": perm,
    "pinter": pinter,
    "powell": powell,
    "qing": qing,
    "quintic": quintic,
    "ridge": ridge,
    "salomon": salomon,
    "schumer_steiglitz": schumer_steiglitz,
    "schwefel_221": schwefel_221,
    "schwefel_222": schwefel_222,
    "modified_schwefel": modified_schwefel,
    "sphere_2": sphere_2,
    "sphere_3": sphere_3,
    "step": step,
    "step_2": step_2,
    "step_3": step_3,
    "stepint": stepint,
    "trid": trid,
    "weierstrass": weierstrass,
    "whitley": whitley,

    "cec_2022_f01": cec_2022_f01,
    "cec_2022_f02": cec_2022_f02,
    "cec_2022_f03": cec_2022_f03,
    "cec_2022_f04": cec_2022_f04,
    "cec_2022_f05": cec_2022_f05,
    "cec_2022_f06": cec_2022_f06,
    "cec_2022_f07": cec_2022_f07,
    "cec_2022_f08": cec_2022_f08,
    "cec_2022_f09": cec_2022_f09,
    "cec_2022_f10": cec_2022_f10,
    "cec_2022_f11": cec_2022_f11,
    "cec_2022_f12": cec_2022_f12,
}


def list_test_functions():
    return sorted(FUNCTIONS.keys())


def get_test_function(name):
    key = str(name).strip().lower()
    if key not in FUNCTIONS:
        raise KeyError(f"Unknown test function: {name}. Available: {', '.join(list_test_functions())}")
    return FUNCTIONS[key]

