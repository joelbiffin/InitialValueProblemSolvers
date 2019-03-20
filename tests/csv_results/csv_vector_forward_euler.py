
import numpy as np
from math import *
from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver


"""
f = lambda u, t: np.array([
    u[0] + u[1] - t,
    u[0] - u[1]
])

true_value = lambda t: np.array([
    math.exp(t*math.sqrt(2)) + math.exp(-1*t*math.sqrt(2)) + 0.5 + 0.5*t,
    (math.sqrt(2) - 1)*math.exp(t*math.sqrt(2)) - (1 + math.sqrt(2)) * math.exp(-1*t*math.sqrt(2)) - 0.5 + 0.5*t
])
"""

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

slv = ForwardEulerSolver(problem, t_n, step)
slv.solve()

slv.solution.write_to_csv("../../outputs/csv_vector_euler.csv", [0, 1])



