import numpy as np

from math import *
from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import RungeKuttaFourthSolver


f = lambda u, t: np.array([
    3*u[0] - 4*u[1],
    4*u[0] -7*u[1]
])

true_value = lambda t: np.array([
    (2/3.0)*exp(t) + (1/3.0)*exp(-5*t),
    (1/3.0)*exp(t) + (2/3.0)*exp(-5*t)
])

de = ODE(f)

u_0 = np.array([1, 1])
t_0 = 0

step = 0.1
t_n = 2

problem = IVP(de, u_0, t_0)

runge_slv = RungeKuttaFourthSolver(problem, t_n, step)
runge_slv.solve()

forward_comparison = ResultsComparator([runge_slv], true_solution=true_value)
forward_comparison.print_result_graphs()

forward_comparison.compute_global_truncation_errors()
forward_comparison.graph_global_truncation_errors()
