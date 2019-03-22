from math import *
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import RungeKuttaFourthSolver, BackwardEulerSolver
from src.multi_step_solvers import AdamsBashforthTwoSolver, AdamsMoultonTwoSolver

from src.predictor_corrector_solvers import PredictorCorrectorSolver

h = lambda u, t: np.array([-2*u[0] + cos(t)])
true_value = lambda t: exp(-2*t) + (1/5.0)*(2*cos(t) + sin(t))

de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.25
t_n = 20


problem = IVP(de, u_0, t_0)

first_step_explicit_slv = RungeKuttaFourthSolver(problem, t_n, step)
first_step_implicit_slv = BackwardEulerSolver(problem, t_n, step)

pred_slv = AdamsBashforthTwoSolver(problem, first_step_explicit_slv, t_n, step)
corr_slv = AdamsMoultonTwoSolver(problem, first_step_implicit_slv, t_n, step)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv, adaptive=False)
pred_corr_slv.solve()

comparison = ResultsComparator([pred_corr_slv], true_solution=true_value)
comparison.print_result_graphs()



