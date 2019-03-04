import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import RungeKuttaFourthSolver


g = lambda u, t: np.array([math.cos(t)])

de = ODE(g)

u_0 = np.array([0])
t_0 = 0

step = 1
precision = 0
t_n = 10


problem = IVP(de, u_0, t_0)

runge_slv = RungeKuttaFourthSolver(problem, t_n, step, precision)
runge_slv.solve()


def true_value(t):
    return math.sin(t)


runge_slv.print_solution()

forward_comparison = ResultsComparator(runge_slv.solution, true_value)
forward_comparison.print_result_graphs()


