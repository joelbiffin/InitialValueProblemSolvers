import numpy as np

from math import *
from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import BackwardEulerSolver


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

slv = BackwardEulerSolver(problem, t_n, step)
slv.solve()

comparison = ResultsComparator([slv], true_solution=true_value)
comparison.print_result_graphs()


comparison.compute_global_truncation_errors()
comparison.graph_global_truncation_errors()


