# pyMetaheuristic

## Introduction

A python library for the following Metaheuristics: **Adaptive Random Search**, **Ant Lion Optimizer**, **Arithmetic Optimization Algorithm**, **Artificial Bee Colony Optimization**, **Artificial Fish Swarm Algorithm**, **Bat Algorithm**, **Biogeography Based Optimization**, **Cross-Entropy Method**, **Crow Search Algorithm**, **Cuckoo Search**, **Differential Evolution**, **Dispersive Flies Optimization**, **Dragonfly Algorithm**, **Firefly Algorithm**, **Flow Direction Algorithm**, **Flower Pollination Algorithm**, **Genetic Algorithm**, **Grasshopper Optimization Algorithm**, **Gravitational Search Algorithm**, **Grey Wolf Optimizer**, **Harris Hawks Optimization**, **Improved Grey Wolf Optimizer**, **Improved Whale Optimization Algorithm**, **Jaya**, **Jellyfish Search Optimizer**, **Krill Herd Algorithm**, **Memetic Algorithm**, **Moth Flame Optimization**, **Multiverse Optimizer**, **Pathfinder Algorithm**, **Particle Swarm Optimization**, **Random Search**, **Salp Swarm Algorithm**, **Simulated Annealing**, **Sine Cosine Algorithm**, **Student Psychology Based Optimization**; **Symbiotic Organisms Search**; **Teaching Learning Based Optimization**, **Whale Optimization Algorithm**.

## Usage

1. Install

```bash
pip install pyMetaheuristic
```

2. Import

```py3

# Import PSO
from pyMetaheuristic.algorithm import particle_swarm_optimization

# Import a Test Function. Available Test Functions: https://bit.ly/3KyluPp
from pyMetaheuristic.test_function import easom

# OR Define your Own Custom Function. The function input should be a list of values, 
# each value represents a dimenstion (x1, x2, ...xn) of the problem.
import numpy as np
def easom(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -np.cos(x1)*np.cos(x2)*np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)
    return func_value

# Run PSO
parameters = {
    'swarm_size': 250,
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'iterations': 500,
    'decay': 0,
    'w': 0.9,
    'c1': 2,
    'c2': 2
}
pso = particle_swarm_optimization(target_function = easom, **parameters)

# Print Solution
variables = pso[:-1]
minimum   = pso[ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

# Plot Solution
from pyMetaheuristic.utils import graphs
plot_parameters = {
    'min_values': (-5, -5),
    'max_values': (5, 5),
    'step': (0.1, 0.1),
    'solution': [variables],
    'proj_view': '3D',
    'view': 'browser'
}
graphs.plot_single_function(target_function = easom, **plot_parameters)

```

3. Colab Demo

Try it in **Colab**:

- Adaptive Random Search ([ Colab Demo ](https://colab.research.google.com/drive/1PbIjDVGAU75Dgxn6I3bpoWovvYA4RYks?usp=sharing)) ([ Original Paper ](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.1623&rep=rep1&type=pdf))
- Ant Lion Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/11GWyd-o11nzwjafF37YDbReAJyjV4Zhp?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.advengsoft.2015.01.010))
- Arithmetic Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1AH0B21_fhF4mOV5iR5MJt_JoUslYE_dt?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.cma.2020.113609))
- Artificial Bee Colony Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1IBouxcnhbNLfCoCV5ueNCq0FZBd9E2gu?usp=sharing)) ( [ Original Paper ](https://abc.erciyes.edu.tr/pub/tr06_2005.pdf))
- Artificial Fish Swarm Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1OugZdsHhg2HQXMryx4AlH3-RdjjeEKlL?usp=sharing)) ( [ Original Paper ](https://www.sysengi.com/EN/10.12011/1000-6788(2002)11-32))
- Bat Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1vbUWQ3T8B1XhPrewaFUW9uvCMGmzajk1?usp=sharing)) ( [ Original Paper ](https://arxiv.org/abs/1004.4170))
- Biogeography Based Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1k3wUNl2R486rkxUhTcTum3usc9f585p0?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/TEVC.2008.919004))
- Cross-Entropy Method ([ Colab Demo ](https://colab.research.google.com/drive/1tI1YbjbAV_O9TdXWYfu8aAlvadC7Crm_?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/S0377-2217(96)00385-2))
- Crow Search Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/18pFLXYi5s9dMgtA03i5yKeC5WZstDp82?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.compstruc.2016.03.001))
- Cuckoo Search ([ Colab Demo ](https://colab.research.google.com/drive/1L1STGmVK5IgdjLpEb-o8tuJ0yPCZ65Mt?usp=sharing)) ( [ Original Paper ](https://arxiv.org/abs/1003.1594v1))
- Differential Evolution ([ Colab Demo ](https://colab.research.google.com/drive/1J56NxxplPOty9rjKQoo5TqN6MzmiqfBe?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1023%2FA%3A1008202821328))
- Dispersive Flies Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1Y6eULdzLMnM2QpApdvABotxwG01BusmE?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.15439/2014F142))
- Dragonfly Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/19xgEwfzdI-yjFMM3e16PbVF1vX8ohu9c?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1007/s00521-020-04866-y))
- Firefly Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1vjUDRdRKPAGo6fTXAsvF9INJiF-wb6Pe?usp=sharing)) ( [ Original Paper ](https://www.sciencedirect.com/book/9780124167438/nature-inspired-optimization-algorithms))
- Flow Direction Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1b72tXxS1X8ntCduN5lUn-An1REcJqp48?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.cie.2021.107224))
- Flower Pollination Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1U7gTgWwBPOWGyEQGX38nSBnBzb3WWAM1?usp=sharing)) ( [ Original Paper ](https://www.sciencedirect.com/book/9780124167438/nature-inspired-optimization-algorithms))
- Genetic Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1zY4N9Sf6odAd1hn8Z3SSww403aj2BHhh?usp=sharing)) ( [ Original Paper ](https://ieeexplore.ieee.org/book/6267401))
- Grey Wolf Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1EQqLtVs9ghQ9Cu-aFRh13hu5ZdgOf9sc?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.advengsoft.2013.12.007))
- Grasshopper Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Mift_Q38gvTkW6eYdkzSS6GpYZKGTwmy?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.advengsoft.2017.01.004))
- Gravitational Search Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1swxMC2Lu9nhObGv7UO5v7eTUm9ULz79Z?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.ins.2009.03.004))
- Harris Hawks Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1swYF7A0I67zX7NxXRJ1d1k1apeMWX2ix?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.future.2019.02.028))
- Improved Grey Wolf Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1Ggu6bd6-FQkLMIrfJynF54b7JBUJaw8Z?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.eswa.2020.113917))
- Improved Whale Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Nvuz7VEqUfUqNzEm1h2_hGhieSH3vgHY?usp=sharing))  ( [ Original Paper ](https://doi.org/10.1016/j.jcde.2019.02.002))
- Jaya ([ Colab Demo ](https://colab.research.google.com/drive/1B-1I3izW0R41_gSGjU26OGHSmy5BY4Tr?usp=sharing)) ( [ Original Paper ](http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf))
- Jellyfish Search Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1yKkUozjzzia9W1sa8XJRNhZzFWCkcGl1?usp=sharing)) ( [ Original Paper ]( https://doi.org/10.1016/j.amc.2020.125535))
- Krill Herd Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1IPQHgHKwR7ELb9EQ--keKmIVrjJLIhZF?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.asoc.2016.08.041))
- Memetic Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1ivRQVK8auSmU9jF3H7CYmpKLlxRHHrPd?usp=sharing)) ( [ Original Paper ](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.9474&rep=rep1&type=pdf))
- Moth Flame Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1-parlgNJ6urQGmNLLViGxf65PhuAS3L4?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.knosys.2015.07.006))
- Multiverse Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1Qna0EHucTYRt9pCfDFzpk9uuNM9tSNKi?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1007/s00521-015-1870-7))
- Pathfinder Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1gntm149Ye1v_vr--zzBCej_5D68SyBHG?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.asoc.2019.03.012))
- Particle Swarm Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1bWAmKTkNKSiSQPUcRdokLQYuhQBOhckZ?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/ICNN.1995.488968))
- Random Search ([ Colab Demo ](https://colab.research.google.com/drive/1DCi4aiO_ORlRq9MetZcxHyKAywMuFkRO?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1080/01621459.1953.10501200))
- Salp Swarm Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Qhkn2NPO5Gavc6ZHW79n_DjmEFeDvOBq?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.advengsoft.2017.07.002))
- Simulated Annealing ([ Colab Demo ](https://colab.research.google.com/drive/1W6X_kCSGOKEDWIJ-ar25kgWIQAc4U1mA?usp=sharing)) ( [ Original Paper ](https://www.jstor.org/stable/1690046))
- Sine Cosine Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1WjbCiks_E2s1qw9l9OkZ4mRQPQuWWYzs?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.knosys.2015.12.022))
- Student Psychology Based Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1T_vFWdPT1qPldVHDTiyMhPiE3YEV9U4j?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.advengsoft.2020.102804))
- Symbiotic Organisms Search ([ Colab Demo ](https://colab.research.google.com/drive/1mvrvi7Q8S1XHKeLCYtZDma9Q48nBewQB?usp=sharing)) ( [ Original Paper ]())
- Teaching Learning Based Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1ulyyREv0K3xPAtBeUdcKXznTzpKrTyL5?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.1016/j.compstruc.2014.03.007))
- Whale Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Nt52dS0AsXm7RHVIt3K0DAaC1i8zKUUC?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.advengsoft.2016.01.008))

4. Test Functions

- Available Test Functions: https://bit.ly/3KyluPp
- Test Functions and their Optimal Solutions with 2D or 3D plots ([ Colab Demo ](https://colab.research.google.com/drive/1Trk6_9IFdBIzZXoVAj2kOBeYJauC5wI8?usp=sharing)) 

# Multiobjective Optimization or Many Objectives Optimization
For Multiobjective Optimization or Many Objectives Optimization try [pyMultiobjective](https://github.com/Valdecy/pyMultiobjective)

# TSP (Travelling Salesman Problem)
For Travelling Salesman Problems try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)

# Acknowledgement 

This section is dedicated to all the people that helped to improve or correct the code. Thank you very much!

* Raiser (01.MARCH.2022) - https://github.com/mpraiser - University of Chinese Academy of Sciences (China)
