import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import AdamsBashforthSecondSolver, ForwardEulerSolver

g = lambda u, t: np.array([math.cos(t)])

de = ODE(g)

u_0 = np.array([0])
t_0 = 0

step = math.pi / 4.0
precision = 10
t_n = 5 * math.pi


problem = IVP(de, u_0, t_0)

first_step_slv = ForwardEulerSolver(problem, t_n, step, precision)


adams_slv = AdamsBashforthSecondSolver(problem, first_step_slv, t_n, step, precision)
adams_slv.solve()



def true_value(t):
    return math.sin(t)

adams_slv.print_solution()

forward_comparison = ResultsComparator(adams_slv.solution, true_value)
forward_comparison.print_result_graphs()


