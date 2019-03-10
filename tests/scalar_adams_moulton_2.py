import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver
from src.multi_step_solvers import AdamsMoultonSecondSolver

g = lambda u, t: np.array([-1*u[0]])

h = lambda u, t: np.array([-2*u[0] + math.cos(t)])
true_value = lambda t: math.exp(-2*t) + (1/5.0)*(2*math.cos(t) + math.sin(t))


de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.4
precision = 3
t_n = 10


problem = IVP(de, u_0, t_0)

first_step_slv = ForwardEulerSolver(problem, t_n, step, precision)


adams_slv = AdamsMoultonSecondSolver(problem, first_step_slv, t_n, step, precision)
adams_slv.solve()



# def true_value(t):
#    return math.exp(-1*t)

adams_slv.print_solution()

forward_comparison = ResultsComparator(adams_slv.solution, true_value)
forward_comparison.print_result_graphs()

forward_comparison.compute_local_truncation_errors()
forward_comparison.graph_local_truncation_errors()

