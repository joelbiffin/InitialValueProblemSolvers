import math
import numpy as np
import time


from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import BackwardEulerSolver


f = lambda u, t: np.array([
    u[0] + u[1] - t,
    u[0] - u[1]
])

true_value = lambda t: np.array([
    math.exp(t*math.sqrt(2)) + math.exp(-1*t*math.sqrt(2)) + 0.5 + 0.5*t,
    (math.sqrt(2) - 1)*math.exp(t*math.sqrt(2)) - (1 + math.sqrt(2)) * math.exp(-1*t*math.sqrt(2)) - 0.5 + 0.5*t
])

de = ODE(f)

u_0 = np.array([2.5, -2.5])
t_0 = 0

step = 0.25
precision = 2
t_n = 3


problem = IVP(de, u_0, t_0)

slv = BackwardEulerSolver(problem, t_n, step, precision)
slv.solve()

comparison = ResultsComparator(slv.solution, true_value)
comparison.print_result_graphs()


comparison.compute_local_truncation_errors()
comparison.graph_local_truncation_errors()


