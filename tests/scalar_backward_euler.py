import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import BackwardEulerSolver


g = lambda u, t: np.array([-1*u[0]])
"""
h = lambda u, t: np.array([-2*u[0] + math.cos(t)])
true_value = lambda t: math.exp(-2*t) + (1/5.0)*(2*math.cos(t) + math.sin(t))
"""

h = lambda u, t: np.array([1-t])
true_value = lambda t: t - 0.5*t*t


de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.5
t_n = 10


problem = IVP(de, u_0, t_0)

slv = BackwardEulerSolver(problem, t_n, step)

slv.solve()




comparison = ResultsComparator([slv], true_solution=true_value)
comparison.print_result_graphs()


comparison.compute_global_truncation_errors()
comparison.graph_global_truncation_errors()
