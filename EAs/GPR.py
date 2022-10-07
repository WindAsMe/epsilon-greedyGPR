from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK
import numpy as np
from cmaes import CMA


def greedy_acquisition(data, label):
    limit_scale, mean = scales_mean(data)
    gpr = GPR(data, label)
    solution = CMAES_tool(mean, 100, data, label, gpr, limit_scale)
    return list(solution)


def scales_mean(data):
    data = np.array(data)
    limit_scale = []
    mean = []
    for i in range(len(data[0])):
        d = data[:, i]
        limit_scale.append([min(d), max(d)])
        mean.append(np.mean(d))
    return limit_scale, mean


def GPR(data, label):
    mixed_kernel = CK(1.0, (1e-4, 1e4)) * RBF(10, (1e-4, 1e4))
    gpr = GaussianProcessRegressor(alpha=5, n_restarts_optimizer=20, kernel=mixed_kernel)
    gpr.fit(data, label)
    return gpr


def CMAES_tool(mean, max_iter, data, label, model, scale_range):
    optimizer = CMA(mean=np.array(mean), bounds=np.array(scale_range), sigma=1.3, population_size=len(data))
    i = np.argmin(label)
    current_best = label[i]
    best_solution = data[i]

    for generation in range(max_iter):
        solutions = []
        Obj = []
        indis = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            obj = model.predict(np.array([x]))
            solutions.append((x, obj))
            indis.append(x)
            Obj.append(obj)
        optimizer.tell(solutions)
        index = np.argmin(Obj)
        if Obj[index] < current_best:
            current_best = Obj[index]
            best_solution = indis[index]
    return best_solution
