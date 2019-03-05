import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import RungeKuttaFourthSolver


g = lambda u, t: np.array([-1*u[0]])
h = lambda u, t: np.array([-2*u[0] + math.cos(t)])
true_value = lambda t: math.exp(-2*t) + (1/5.0)*(2*math.cos(t) + math.sin(t))



de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.5
precision = 1
t_n = 10


problem = IVP(de, u_0, t_0)

runge_slv = RungeKuttaFourthSolver(problem, t_n, step, precision)
runge_slv.solve()


# def true_value(t):
#     return math.exp(-1*t)


runge_slv.print_solution()

forward_comparison = ResultsComparator(runge_slv.solution, true_value)
forward_comparison.print_result_graphs()


