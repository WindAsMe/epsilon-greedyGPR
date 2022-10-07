import numpy as np
from cmaes import CMA
from EAs.GPR import greedy_acquisition


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
        elite = greedy_acquisition(indis, Obj)
        elite = np.array(elite, "double")
        elite_f = func(elite)
        i = np.argmax(Obj)
        if Obj[i] > elite_f:
            Obj[i] = elite_f
            solutions[i] = (elite, elite_f)
        optimizer.tell(solutions)
        current_best = min(current_best, min(Obj))
    return current_best








