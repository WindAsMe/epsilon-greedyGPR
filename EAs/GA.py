import geatpy as ea
from Problems import MyProblem


def GA_exe(Dim, max_iter, NIND, func, scale_range):
    obj_trace = []
    problem = MyProblem.problem(Dim, func, scale_range, obj_trace)  # 实例化问题对象
    population = ea.Population(Encoding="RI", NIND=NIND)
    """===========================算法参数设置=========================="""
    myAlgorithm = ea.soea_EGA_templet(problem, population)
    myAlgorithm.MAXGEN = max_iter
    myAlgorithm.drawing = 0
    """=====================调用算法模板进行种群进化====================="""
    solution = ea.optimize(myAlgorithm, verbose=False, outputMsg=False, drawLog=False, saveFlag=False)
    return solution['ObjV'][0][0]

