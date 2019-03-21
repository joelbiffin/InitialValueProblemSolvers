import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver
from src.predictor_corrector_solvers import PredictorCorrectorSolver

# h = lambda u, t: np.array([-2*u[0] + math.cos(t)])
# true_value = lambda t: math.exp(-2*t) + (1/5.0)*(2*math.cos(t) + math.sin(t))

h = lambda u, t: np.array([-2*u[0] + math.cos(t)])
true_value = lambda t: math.exp(-2*t) + (1/5.0)*(2*math.cos(t) + math.sin(t))

de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.25
t_n = 20

def local_truncation_error_estimate(prediction, correction, time_mesh=None, this_step=None):
    lte_vec = np.zeros_like(prediction)
    for i, p in enumerate(prediction):
        lte_vec[i] = math.fabs(p - correction[i])

    return lte_vec


problem = IVP(de, u_0, t_0)

pred_slv = ForwardEulerSolver(problem, t_n, step, step_tol=1e-3)
corr_slv = BackwardEulerSolver(problem, t_n, step)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv, lte=local_truncation_error_estimate)
pred_corr_slv.solve()
print("{}.solve() ran in {}s.".format(str(pred_corr_slv), pred_corr_slv.solve_time))


forward_slv = ForwardEulerSolver(problem, t_n, step)
forward_slv.solve()
print("{}.solve() ran in {}s.".format(str(forward_slv), forward_slv.solve_time))

backward_slv = BackwardEulerSolver(problem, t_n, step)
backward_slv.solve()
print("{}.solve() ran in {}s.".format(str(backward_slv), backward_slv.solve_time))


comparison = ResultsComparator([pred_corr_slv], true_solution=true_value)
comparison.print_result_graphs()

comparison = ResultsComparator([pred_corr_slv, forward_slv, backward_slv], true_solution=true_value)
comparison.print_result_graphs()

comparison.graph_local_truncation_errors()
