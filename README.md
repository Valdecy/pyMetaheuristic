# pyMetaheuristic

## Introduction

A python library for the following Metaheuristics: **Adaptive Random Search**, **Ant Lion Optimizer**, **Arithmetic Optimization Algorithm**, **Artificial Bee Colony Optimization**, **Bat Algorithm**, **Biogeography Based Optimization**, **Cross-Entropy Method**, **Cuckoo Search**, **Differential Evolution**, **Dispersive Flies Optimization**, **Dragonfly Algorithm**, **Firefly Algorithm**, **Flow Direction Algorithm**, **Flower Pollination Algorithm**, **Genetic Algorithm**, **Gravitational Search Algorithm**, **Grey Wolf Optimizer**, **Harris Hawks Optimization**, **Improved Grey Wolf Optimizer**, **Improved Whale Optimization Algorithm**, **Jaya**, **Memetic Algorithm**, **Moth Flame Optimization**, **Multiverse Optimizer**, **Particle Swarm Optimization**, **Random Search**, **Salp Swarm Algorithm**, **Simulated Annealing**, **Sine Cosine Algorithm**, **Whale Optimization Algorithm**.

## Usage

1. Install
```bash
pip install pyMetaheuristic
```

2. Import

```py3

# Import PSO
from pyMetaheuristic.algorithm import particle_swarm_optimization

# Import Test Function. Available Test Functions: easom, six_hump_camel_back, de_jong_1, rosenbrocks_valley, axis_parallel_hyper_ellipsoid, rastrigin, branin_rcos, goldstein_price
from pyMetaheuristic.test_function import easom

# OR Define your Own Custom Function
import numpy as np
def easom(variables_values = [0, 0]):
    x1, x2     = variables_values
    func_value = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - pi) ** 2)
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

# Show Solution
variables = pso[:-1]
minimum   = pso[ -1]
print('Variables: ', np.around(variables, 4) , ' Minimum Value Found: ', round(minimum, 4) )

```

3. Colab Demo

Try it in **Colab**:

- Adaptive Random Search ([ Colab Demo ](https://colab.research.google.com/drive/1PbIjDVGAU75Dgxn6I3bpoWovvYA4RYks?usp=sharing))
- Ant Lion Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/11GWyd-o11nzwjafF37YDbReAJyjV4Zhp?usp=sharing))
- Arithmetic Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1AH0B21_fhF4mOV5iR5MJt_JoUslYE_dt?usp=sharing))
- Artificial Bee Colony Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1IBouxcnhbNLfCoCV5ueNCq0FZBd9E2gu?usp=sharing))
- Bat Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1vbUWQ3T8B1XhPrewaFUW9uvCMGmzajk1?usp=sharing))
- Biogeography Based Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1k3wUNl2R486rkxUhTcTum3usc9f585p0?usp=sharing))
- Cross-Entropy Method ([ Colab Demo ](https://colab.research.google.com/drive/1tI1YbjbAV_O9TdXWYfu8aAlvadC7Crm_?usp=sharing))
- Cuckoo Search ([ Colab Demo ](https://colab.research.google.com/drive/1L1STGmVK5IgdjLpEb-o8tuJ0yPCZ65Mt?usp=sharing))
- Differential Evolution ([ Colab Demo ](https://colab.research.google.com/drive/1J56NxxplPOty9rjKQoo5TqN6MzmiqfBe?usp=sharing))
- Dispersive Flies Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1Y6eULdzLMnM2QpApdvABotxwG01BusmE?usp=sharing))
- Dragonfly Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/19xgEwfzdI-yjFMM3e16PbVF1vX8ohu9c?usp=sharing))
- Firefly Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1vjUDRdRKPAGo6fTXAsvF9INJiF-wb6Pe?usp=sharing))
- Flow Direction Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1b72tXxS1X8ntCduN5lUn-An1REcJqp48?usp=sharing))
- Flower Pollination Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1U7gTgWwBPOWGyEQGX38nSBnBzb3WWAM1?usp=sharing))
- Genetic Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1zY4N9Sf6odAd1hn8Z3SSww403aj2BHhh?usp=sharing))
- Grey Wolf Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1EQqLtVs9ghQ9Cu-aFRh13hu5ZdgOf9sc?usp=sharing))
- Gravitational Search Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1swxMC2Lu9nhObGv7UO5v7eTUm9ULz79Z?usp=sharing))
- Harris Hawks Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1swYF7A0I67zX7NxXRJ1d1k1apeMWX2ix?usp=sharing))
- Improved Grey Wolf Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1Ggu6bd6-FQkLMIrfJynF54b7JBUJaw8Z?usp=sharing))
- Improved Whale Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Nvuz7VEqUfUqNzEm1h2_hGhieSH3vgHY?usp=sharing))
- Jaya ([ Colab Demo ](https://colab.research.google.com/drive/1B-1I3izW0R41_gSGjU26OGHSmy5BY4Tr?usp=sharing))
- Memetic Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1ivRQVK8auSmU9jF3H7CYmpKLlxRHHrPd?usp=sharing))
- Moth Flame Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1-parlgNJ6urQGmNLLViGxf65PhuAS3L4?usp=sharing))
- Multiverse Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1Qna0EHucTYRt9pCfDFzpk9uuNM9tSNKi?usp=sharing))
- Particle Swarm Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1bWAmKTkNKSiSQPUcRdokLQYuhQBOhckZ?usp=sharing))
- Random Search ([ Colab Demo ](https://colab.research.google.com/drive/1DCi4aiO_ORlRq9MetZcxHyKAywMuFkRO?usp=sharing))
- Salp Swarm Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Qhkn2NPO5Gavc6ZHW79n_DjmEFeDvOBq?usp=sharing))
- Simulated Annealing ([ Colab Demo ](https://colab.research.google.com/drive/1W6X_kCSGOKEDWIJ-ar25kgWIQAc4U1mA?usp=sharing))
- Sine Cosine Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1WjbCiks_E2s1qw9l9OkZ4mRQPQuWWYzs?usp=sharing))
- Whale Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Nt52dS0AsXm7RHVIt3K0DAaC1i8zKUUC?usp=sharing))
