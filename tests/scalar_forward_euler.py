import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver

g = lambda u, t: np.array([math.cos(t)])
true_value = lambda t: np.array([math.sin(t)])

de = ODE(g)

u_0 = np.array([0])
t_0 = 0

step = 0.01
t_n = 12


problem = IVP(de, u_0, t_0)

slv = ForwardEulerSolver(problem, t_n, step)
slv.solve()

print(".solve() ran in {}s.".format(slv.solve_time))


comparison = ResultsComparator([slv], true_solution=true_value)
comparison.print_result_graphs()

comparison.compute_global_truncation_errors()
comparison.graph_global_truncation_errors()


