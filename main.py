from cec2013single.cec2013 import Benchmark
import numpy as np
from os import path
from EAs import DE, GA, CMAES, hybridDE, hybridGA, hybridCMAES


def write_obj(data, path):
    with open(path, 'a') as f:
        f.write(str(data) + ', ')
        f.close()


if __name__ == "__main__":
    Dims = [2, 10, 30]
    bench = Benchmark()
    this_path = path.realpath(__file__)

    trial = 1
    lb, ub = -100, 100
    for func_num in range(1, 29):
        for Dim in Dims:
            NIND = 50 * Dim
            FEs = 1000 * Dim
            Max_iter = int(FEs / NIND)
            scale_range = []
            for i in range(Dim):
                scale_range.append([lb, ub])
            for run in range(trial):
                CMAES_obj_path = path.dirname(this_path) + '/data/CMAES/' + str(Dim) + 'D/f' + str(func_num)
                DE_obj_path = path.dirname(this_path) + '/data/DE/' + str(Dim) + 'D/f' + str(func_num)
                GA_obj_path = path.dirname(this_path) + '/data/GA/' + str(Dim) + 'D/f' + str(func_num)

                hybridCMAES_obj_path = path.dirname(this_path) + '/data/hybridCMAES/' + str(Dim) + 'D/f' + str(func_num)
                hybridDE_obj_path = path.dirname(this_path) + '/data/hybridDE/' + str(Dim) + 'D/f' + str(func_num)
                hybridGA_obj_path = path.dirname(this_path) + '/data/hybridGA/' + str(Dim) + 'D/f' + str(func_num)

                func = bench.get_function(func_num)

                CMAES_obj = CMAES.CMAES_exe(Dim, Max_iter, NIND, func, scale_range)
                write_obj(CMAES_obj, CMAES_obj_path)

                DE_obj = DE.DE_exe(Dim, Max_iter, NIND, func, scale_range)
                write_obj(DE_obj, DE_obj_path)

                GA_obj = GA.GA_exe(Dim, Max_iter, NIND, func, scale_range)
                write_obj(GA_obj, GA_obj_path)

                """hybrid CMA-ES, DE, GA"""
                hybridCMAES_obj = hybridCMAES.CMAES_exe(Dim, Max_iter, NIND, func, scale_range)
                write_obj(hybridCMAES_obj, hybridCMAES_obj_path)

                hybridDE_obj = hybridDE.DE_exe(Dim, Max_iter, NIND, func, scale_range)
                write_obj(hybridDE_obj, hybridDE_obj_path)

                hybridGA_obj = hybridGA.GA_exe(Dim, Max_iter, NIND, func, scale_range)
                write_obj(hybridGA_obj, hybridGA_obj_path)

