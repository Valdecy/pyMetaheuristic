# pyMetaheuristic

## Introduction

A python library for the following Metaheuristics: **Adaptive Random Search**, **Ant Lion Optimizer**, **Arithmetic Optimization Algorithm**, **Artificial Bee Colony Optimization**, **Artificial Fish Swarm Algorithm**, **Bat Algorithm**, **Biogeography Based Optimization**, **Cat Swarm Optimization**, **Chicken Swarm Optimization**, **Cockroach Swarm Optimization**, **Cross-Entropy Method**, **Crow Search Algorithm**, **Cuckoo Search**, **Differential Evolution**, **Dispersive Flies Optimization**, **Dragonfly Algorithm**, **Dynamic Virtual Bats Algorithm**, **Elephant Herding Optimization Algorithm**, **Firefly Algorithm**, **Flow Direction Algorithm**, **Flower Pollination Algorithm**, **Genetic Algorithm**, **Grasshopper Optimization Algorithm**, **Gravitational Search Algorithm**, **Grey Wolf Optimizer**, **Harris Hawks Optimization**, **Hunting Search Algorithm**, **Improved Grey Wolf Optimizer**, **Improved Whale Optimization Algorithm**, **Jaya**, **Jellyfish Search Optimizer**, **Krill Herd Algorithm**, **Memetic Algorithm**, **Monarch Butterfly Optimization**, **Moth Flame Optimization**, **Multiverse Optimizer**, **Pathfinder Algorithm**, **Particle Swarm Optimization**, **Random Search**, **Salp Swarm Algorithm**, **Simulated Annealing**, **Sine Cosine Algorithm**, **Student Psychology Based Optimization**; **Symbiotic Organisms Search**; **Teaching Learning Based Optimization**, **Whale Optimization Algorithm**.

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
- Cat Swarm Optimization ([ Colab Demo ](https://colab.research.google.com/drive/16kULfNzZsFayvAf9IYgF-260iWy5x-u4?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1007/978-3-540-36668-3_94))
- Chicken Swarm Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1twWQX1rsZE0zcF36CIdBIvueNH-FfIif?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1007/978-3-319-11857-4_10))
- Cockroach Swarm Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1lrsRWMJhX2Uf-IGObukpZM7t2zr-OPW8?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/ICCET.2010.5485993))
- Cross-Entropy Method ([ Colab Demo ](https://colab.research.google.com/drive/1tI1YbjbAV_O9TdXWYfu8aAlvadC7Crm_?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/S0377-2217(96)00385-2))
- Crow Search Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/18pFLXYi5s9dMgtA03i5yKeC5WZstDp82?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.compstruc.2016.03.001))
- Cuckoo Search ([ Colab Demo ](https://colab.research.google.com/drive/1L1STGmVK5IgdjLpEb-o8tuJ0yPCZ65Mt?usp=sharing)) ( [ Original Paper ](https://arxiv.org/abs/1003.1594v1))
- Differential Evolution ([ Colab Demo ](https://colab.research.google.com/drive/1J56NxxplPOty9rjKQoo5TqN6MzmiqfBe?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1023%2FA%3A1008202821328))
- Dispersive Flies Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1Y6eULdzLMnM2QpApdvABotxwG01BusmE?usp=sharing)) ( [ Original Paper ](http://dx.doi.org/10.15439/2014F142))
- Dragonfly Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/19xgEwfzdI-yjFMM3e16PbVF1vX8ohu9c?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1007/s00521-015-1920-1))
- Dynamic Virtual Bats Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1IKyCxK50he9ghhmyLRrTqf-AcD7kllG_?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/INCoS.2014.40))
- Elephant Herding Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1wom7cm23VN0N40_23HsoAKktkMy7V3ts?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/ISCBI.2015.8))
- Firefly Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1vjUDRdRKPAGo6fTXAsvF9INJiF-wb6Pe?usp=sharing)) ( [ Original Paper ](https://www.sciencedirect.com/book/9780124167438/nature-inspired-optimization-algorithms))
- Flow Direction Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1b72tXxS1X8ntCduN5lUn-An1REcJqp48?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.cie.2021.107224))
- Flower Pollination Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1U7gTgWwBPOWGyEQGX38nSBnBzb3WWAM1?usp=sharing)) ( [ Original Paper ](https://www.sciencedirect.com/book/9780124167438/nature-inspired-optimization-algorithms))
- Genetic Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1zY4N9Sf6odAd1hn8Z3SSww403aj2BHhh?usp=sharing)) ( [ Original Paper ](https://ieeexplore.ieee.org/book/6267401))
- Grey Wolf Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1EQqLtVs9ghQ9Cu-aFRh13hu5ZdgOf9sc?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.advengsoft.2013.12.007))
- Grasshopper Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Mift_Q38gvTkW6eYdkzSS6GpYZKGTwmy?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.advengsoft.2017.01.004))
- Gravitational Search Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1swxMC2Lu9nhObGv7UO5v7eTUm9ULz79Z?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.ins.2009.03.004))
- Harris Hawks Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1swYF7A0I67zX7NxXRJ1d1k1apeMWX2ix?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.future.2019.02.028))
- Hunting Search Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1usqrl2Ljoj9ha7wuShD1JgFlHrAP4K0Z?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1109/ICSCCW.2009.5379451))
- Improved Grey Wolf Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1Ggu6bd6-FQkLMIrfJynF54b7JBUJaw8Z?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.eswa.2020.113917))
- Improved Whale Optimization Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1Nvuz7VEqUfUqNzEm1h2_hGhieSH3vgHY?usp=sharing))  ( [ Original Paper ](https://doi.org/10.1016/j.jcde.2019.02.002))
- Jaya ([ Colab Demo ](https://colab.research.google.com/drive/1B-1I3izW0R41_gSGjU26OGHSmy5BY4Tr?usp=sharing)) ( [ Original Paper ](http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_32.pdf))
- Jellyfish Search Optimizer ([ Colab Demo ](https://colab.research.google.com/drive/1yKkUozjzzia9W1sa8XJRNhZzFWCkcGl1?usp=sharing)) ( [ Original Paper ]( https://doi.org/10.1016/j.amc.2020.125535))
- Krill Herd Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1IPQHgHKwR7ELb9EQ--keKmIVrjJLIhZF?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1016/j.asoc.2016.08.041))
- Memetic Algorithm ([ Colab Demo ](https://colab.research.google.com/drive/1ivRQVK8auSmU9jF3H7CYmpKLlxRHHrPd?usp=sharing)) ( [ Original Paper ](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.9474&rep=rep1&type=pdf))
- Monarch Butterfly Optimization ([ Colab Demo ](https://colab.research.google.com/drive/1-th99S0O93gpRbXcFtj1G3DeYP2iDcGP?usp=sharing)) ( [ Original Paper ](https://doi.org/10.1007/s00521-015-1923-y))
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

- Ackley ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/14avAOSIGInxQpvfiwBKKtKOBHxoW23oG?usp=sharing)) 
- Axis Parallel Hyper-Ellipsoid ( [ Paper ](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1t0wZbzZRLhpCxnoik6c7IvSPmRjIKLem?usp=sharing))
- Beale ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1AwoAXEtXrIKCkhT1bvFra2Sh9PX_mT46?usp=sharing))
- Bohachevsky F1 ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1UKYZlBkc85RXVx83JCXa9V9PG6qSsQYc?usp=sharing))
- Bohachevsky F2 ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1lvjgOu7ON3Z12RyxKXgpv90YkjPBmfRK?usp=sharing))
- Bohachevsky F3 ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/18iMq9-XgiCMCKCEbZmuviFzjbYXIBBmC?usp=sharing))
- Booth ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1q2H0lvQqLuKUT9scURWA4CKqDXLR8tp_?usp=sharing))
- Branin RCOS ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1Zz8xhZRlxgjvF8SzemFWEFfU7DOPecab?usp=sharing))
- Bukin F6 ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1oFRyahRak54c0UFOZ3RwPCYFzYU6vc2g?usp=sharing))
- Cross in Tray ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/14wf2skMXUuGCnvOs5TTpWHhofp8t-gMP?usp=sharing))
- De Jong F1 ( [ Paper ](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1Oiz5VtRYgvioW914IxVLNgjDbWK5cOse?usp=sharing))
- Drop Wave ( [ Paper ](https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1Z7XDYXuKc6rSGGpzZBzAO1G9N6866QyM?usp=sharing))
- Easom ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1AYFtE5H4QtitHgXiAuOQHUmeNShpGlRM?usp=sharing))
- Eggholder ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1nkxsKKyAeXqhDyDUMoRvTike8WagN9QT?usp=sharing))
- Goldstein-Price ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1XIpaYfD5VT_RMgt2c_6APMItxdz-x629?usp=sharing))
- Griewangk F8 ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1sg9W5zwDvNk0s_2ZHHxlME4XXjuBDO1C?usp=sharing))
- Himmelblau ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1K5BEs3iP56YblVkLTtR7ONPH42Ir3TdX?usp=sharing))
- Holder Table ( [ Paper ](https://mpra.ub.uni-muenchen.de/2718/1/MPRA_paper_2718.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1lBAxNnUeBeiSUeQFKg8aNfSUDnRNc-_6?usp=sharing))
- Matyas ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1HeD0EPFAr1psHEuDGZjqIJ3eznP1zasN?usp=sharing))
- McCormick ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1w3FPlw_09mwagLyY6_0eU90IE_9_I1Af?usp=sharing))
- Levi F13 ( [ Paper ](https://mpra.ub.uni-muenchen.de/2718/1/MPRA_paper_2718.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1lFymXZfR9g02eVnJGa_9NvDh6wK5u3FI?usp=sharing))
- Rastrigin ( [ Paper ](https://doi.org/10.1007/978-3-031-14721-0_35)) ( [ Plot ](https://colab.research.google.com/drive/1HNcRovhz9VnH9r98VNCEzfORHkmVGhOy?usp=sharing))
- Rosenbrocks Valley (De Jong F2) ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1OAgEPn98g_3EegI6GpwNSOoGg3gBMlmC?usp=sharing))
- Schaffer F2 ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1wEKPcUg4_GgF5IRvbHpsnIVnwEJg9Mmm?usp=sharing))
- Schaffer F4 ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/188bzwrUUozIMrLZsaMaouyc7M1OxzHJH?usp=sharing))
- Schaffer F6 ( [ Paper ](http://dx.doi.org/10.1016/j.cam.2017.04.047)) ( [ Plot ](https://colab.research.google.com/drive/1HYdtuQoo4IgBwa4h7Pr83PcLHhyMJ7AK?usp=sharing))
- Schwefel ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1Ebq_c0HM13tGdpCSCWqHgTpCq0wOOgzV?usp=sharing))
- Six Hump Camel Back ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1pxGLC7W0MGvVjjkuGmYTTayb0EbXrfe_?usp=sharing))
- Styblinski-Tang ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1-90p9zL3oQWxo2VONKNd5cVZX736oyif?usp=sharing))
- Three Hump Camel Back ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1CeZ94mf32Ql5ommM1uWk3bEwOvsEiP2R?usp=sharing))
- Zakharov ( [ Paper ](https://arxiv.org/pdf/1308.4008.pdf)) ( [ Plot ](https://colab.research.google.com/drive/1XmnduTRcIK6aTEeAnSzbbJ8uK9c-KpJi?usp=sharing))

# Multiobjective Optimization or Many Objectives Optimization
For Multiobjective Optimization or Many Objectives Optimization try [pyMultiobjective](https://github.com/Valdecy/pyMultiobjective)

# TSP (Travelling Salesman Problem)
For Travelling Salesman Problems try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)

# Acknowledgement 

This section is dedicated to all the people that helped to improve or correct the code. Thank you very much!

* Raiser (01.MARCH.2022) - https://github.com/mpraiser - University of Chinese Academy of Sciences (China)
