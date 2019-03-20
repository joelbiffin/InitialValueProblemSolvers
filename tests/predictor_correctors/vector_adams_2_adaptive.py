import numpy as np

from math import *
from src.ivp import IVP
from src.ode import ODE
from src.predictor_corrector_solvers import PredictorCorrectorSolver
from src.results import ResultsComparator
from src.one_step_solvers import RungeKuttaFourthSolver, BackwardEulerSolver
from src.multi_step_solvers import AdamsMoultonTwoSolver, AdamsBashforthTwoSolver

f = lambda u, t: np.array([
    3*u[0] - 4*u[1],
    4*u[0] - 7*u[1]
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



def adapt_step(stepped, lte, tolerance):
    print(tolerance)
    update = stepped * pow((tolerance / (np.linalg.norm(lte, 2))), (1.0/3))

    if update > (stepped * 2):
        return 2 * stepped
    elif (2 * update) < stepped:
        return 0.5 * stepped

    return update


def local_truncation_error_estimate(prediction, correction, time_mesh, this_step):
    lte_vec = np.zeros_like(prediction)
    for i, p in enumerate(prediction):
        lte_vec[i] = fabs(p - correction[i])

    steps = [time_mesh[this_step] - time_mesh[this_step-1],
             time_mesh[this_step - 1] - time_mesh[this_step-2]]

    return lte_vec / (3 + (steps[1] / steps[0]))


problem = IVP(de, u_0, t_0)

first_step_explicit_slv = RungeKuttaFourthSolver(problem, t_n, step)
first_step_implicit_slv = BackwardEulerSolver(problem, t_n, step)

pred_slv = AdamsBashforthTwoSolver(problem, first_step_explicit_slv, t_n, step)
corr_slv = AdamsMoultonTwoSolver(problem, first_step_implicit_slv, t_n, step)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv, adaptive=True,
                                         method=adapt_step, lte=local_truncation_error_estimate)
pred_corr_slv.solve()


comparison = ResultsComparator([pred_corr_slv], true_solution=true_value)
comparison.print_result_graphs()
