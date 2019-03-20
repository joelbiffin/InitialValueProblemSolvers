import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.one_step_solvers import ForwardEulerSolver

g = lambda u, t: np.array([math.cos(t)])
true_value = lambda t: np.array([math.sin(t)])

de = ODE(g)

u_0 = np.array([0])
t_0 = 0

step = 1
t_n = 12

problem = IVP(de, u_0, t_0)

slv = ForwardEulerSolver(problem, t_n, step)
slv.solve()

slv.solution.write_to_csv("../../outputs/csv_sin_euler.csv", [0])



