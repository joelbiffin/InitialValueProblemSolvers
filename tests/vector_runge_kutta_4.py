import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import RungeKuttaFourthSolver


g = lambda u, t: np.array([u[1], (1/2.0)*u[0] + (5/2.0)*u[1]])

de = ODE(g)

u_0 = np.array([6, -1])
t_0 = 3

step = 0.00001
precision = 5
t_n = 5


problem = IVP(de, u_0, t_0)

runge_slv = RungeKuttaFourthSolver(problem, t_n, step, precision)
runge_slv.solve()


def true_value(t):
    return 0


runge_slv.print_solution()

forward_comparison = ResultsComparator(runge_slv.solution, true_value)
forward_comparison.print_result_graphs()


