from math import *
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.predictor_corrector_solvers import PredictorCorrectorSolver
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver


"""
f = lambda u, t: np.array([
    3*u[0] - 4*u[1],
    4*u[0] -7*u[1]
])

true_value = lambda t: np.array([
    (2/3.0)*exp(t) + (1/3.0)*exp(-5*t),
    (1/3.0)*exp(t) + (2/3.0)*exp(-5*t)
])
"""

f = lambda u, t: np.array([
    -3*u[0] - 4*u[1] + 23,
    2*u[0] + u[1] - 7
])

true_value = lambda t: np.array([
    1 + exp(-t) * (7*cos(2*t) + 3*sin(2*t)),
    5 - exp(-t) * (5*cos(2*t) - 2*sin(2*t))
])


de = ODE(f)

u_0 = np.array([8, 0])
t_0 = 0

step = 0.01
t_n = 10


problem = IVP(de, u_0, t_0)

pred_slv = ForwardEulerSolver(problem, t_n, step, step_tol=5e-2)
corr_slv = BackwardEulerSolver(problem, t_n, step)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv, adaptive=False)
pred_corr_slv.solve()


comparison = ResultsComparator([pred_corr_slv], true_solution=true_value, has_lte=True)
comparison.print_result_graphs()

comparison.graph_local_truncation_errors()

