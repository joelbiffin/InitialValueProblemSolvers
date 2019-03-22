import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver
from src.multi_step_solvers import AdamsMoultonTwoSolver
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


problem = IVP(de, u_0, t_0)

first_implicit_slv = BackwardEulerSolver(problem, t_n, step)

pred_slv = ForwardEulerSolver(problem, t_n, step, step_tol=1e-3)
corr_slv = AdamsMoultonTwoSolver(problem, first_implicit_slv, t_n, step)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv)
pred_corr_slv.solve()



comparison = ResultsComparator([pred_corr_slv], true_solution=true_value)
comparison.print_result_graphs()

comparison.graph_local_truncation_errors()
