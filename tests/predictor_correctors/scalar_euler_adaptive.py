from math import *
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver
from src.predictor_corrector_solvers import PredictorCorrectorSolver

h = lambda u, t: np.array([-2*u[0] + cos(t)])
true_value = lambda t: exp(-2*t) + (1/5.0)*(2*cos(t) + sin(t))


de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.3
t_n = pi * 10


def adapt_step(stepped, lte, tolerance):
    update = stepped * pow((tolerance / (np.linalg.norm(lte, 2))), 0.5)

    if update > (stepped * 2):
        return 2 * stepped
    elif (2 * update) < stepped:
        return 0.5 * stepped

    return update


def local_truncation_error_estimate(prediction, correction, time_mesh=None, this_step=None):
    lte_vec = np.zeros_like(prediction)
    for i, p in enumerate(prediction):
        lte_vec[i] = fabs(p - correction[i])

    return lte_vec



problem = IVP(de, u_0, t_0)

pred_slv = ForwardEulerSolver(problem, t_n, step, step_tol=1e-4)
corr_slv = BackwardEulerSolver(problem, t_n, step)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv, adaptive=True, method=adapt_step)
pred_corr_slv.solve()


comparison = ResultsComparator([pred_corr_slv], true_solution=true_value)
comparison.print_result_graphs()

comparison.graph_local_truncation_errors()