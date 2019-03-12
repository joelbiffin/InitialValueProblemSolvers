import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, RungeKuttaFourthSolver
from src.multi_step_solvers import AdamsMoultonSecondSolver

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

step = 0.0001
t_n = 3


problem = IVP(de, u_0, t_0)

first_step_slv = ForwardEulerSolver(problem, t_n, step)
adams_slv = AdamsMoultonSecondSolver(problem, first_step_slv, t_n, step)

adams_slv.solve()


forward_comparison = ResultsComparator(adams_slv.solution, true_value)
forward_comparison.print_result_graphs()

forward_comparison.compute_local_truncation_errors()
forward_comparison.graph_local_truncation_errors()
