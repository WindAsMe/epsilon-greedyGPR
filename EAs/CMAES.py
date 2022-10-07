import numpy as np
from cmaes import CMA


def CMAES_exe(dim, max_iter, NIND, func, scale_range):
    optimizer = CMA(mean=np.zeros(dim), bounds=np.array(scale_range), sigma=1.3, population_size=NIND)
    current_best = float("inf")
    for generation in range(max_iter):
        solutions = []
        Obj = []
        indis = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            obj = func(x)
            solutions.append((x, obj))
            indis.append(x)
            Obj.append(obj)
        optimizer.tell(solutions)
        current_best = min(current_best, min(Obj))
    return current_best








