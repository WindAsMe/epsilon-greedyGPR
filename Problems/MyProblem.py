import geatpy as ea
import numpy as np
from pymoo.core.problem import Problem


class problem(ea.Problem):

    def __init__(self, Dim, func, scale_range, obj_trace):
        name = 'MyProblem'
        M = 1
        maxormins = [1]
        self.Dim = Dim
        varTypes = [0] * self.Dim
        scale_range = np.array(scale_range)
        lb = scale_range[:, 0]
        ub = scale_range[:, 1]
        lbin = [1] * self.Dim
        ubin = [1] * self.Dim
        self.func = func
        self.obj_trace = obj_trace
        self.current_best = float("inf")
        ea.Problem.__init__(self, name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数，pop为传入的种群对象
        Obj = []
        for p in pop.Phen:
            obj = self.func(p)
            Obj.append([obj])
        pop.ObjV = np.array(Obj)
        self.current_best = min(self.current_best, np.min(pop.ObjV))
        self.obj_trace.append(self.current_best)

    def evalVars(self, Vars):
        return self.func(Vars, self.Dim)


"""This problem is specific for PSO and Random Search"""
class pyProblem(Problem):

    def __init__(self, dim, func, obj_trace):
        super().__init__(n_var=dim, n_obj=1, n_constr=0, xl=-100, xu=100)
        self.func = func
        self.obj_trace = obj_trace
        self.current_best = float("inf")

    def _evaluate(self, x, out, *args, **kwargs):
        Obj = []
        for d in x:
            Obj.append([self.func(d, self.n_var)])
        self.current_best = min(self.current_best, np.min(Obj))
        self.obj_trace.append(self.current_best)
        out["F"] = np.array(Obj)


