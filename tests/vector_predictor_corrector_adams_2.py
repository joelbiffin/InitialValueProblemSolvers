import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.predictor_corrector_solvers import PredictorCorrectorSolver
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver
from src.multi_step_solvers import AdamsBashforthThirdSolver, AdamsMoultonSecondSolver

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

step = 0.1
precision = 3
t_n = 6


problem = IVP(de, u_0, t_0)

first_step_slv = ForwardEulerSolver(problem, t_n, step, precision)
pred_slv = AdamsBashforthThirdSolver(problem, first_step_slv, t_n, step, precision)
corr_slv = AdamsMoultonSecondSolver(problem, first_step_slv, t_n, step, precision)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv)
pred_corr_slv.solve()

comparison = ResultsComparator(pred_corr_slv.solution, true_value)
comparison.print_result_graphs()
comparison.compute_local_truncation_errors()
comparison.graph_local_truncation_errors()
