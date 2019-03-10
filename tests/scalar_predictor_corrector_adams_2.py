import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver
from src.multi_step_solvers import AdamsBashforthSecondSolver, AdamsMoultonSecondSolver

from src.predictor_corrector_solvers import PredictorCorrectorSolver

h = lambda u, t: np.array([-2*u[0] + math.cos(t)])
true_value = lambda t: math.exp(-2*t) + (1/5.0)*(2*math.cos(t) + math.sin(t))

de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.251
precision = 3
t_n = 100


problem = IVP(de, u_0, t_0)

first_step_slv = ForwardEulerSolver(problem, t_n, step, precision)
pred_slv = AdamsBashforthSecondSolver(problem, first_step_slv, t_n, step, precision)
corr_slv = AdamsMoultonSecondSolver(problem, first_step_slv, t_n, step, precision)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv)
pred_corr_slv.solve()
pred_corr_slv.print_solution()



comparison = ResultsComparator(pred_corr_slv.solution, true_value)
comparison.print_result_graphs()


