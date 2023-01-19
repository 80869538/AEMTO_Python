import numpy as np
from pymoo.core.population import pop_from_array_or_individual
from pymoo.core.population import Population

def FPS(pop, num_sel):
    F = pop.get("F")
    P = F/F.sum()
    wheel = np.add.accumulate(P)
    selected = []
    for n in range(num_sel):
        r = np.random.uniform(0,1)
        for (i, individual) in enumerate(pop):
            if r <= wheel[i]:
                selected.append(individual)
                break
    return Population(individuals=selected)