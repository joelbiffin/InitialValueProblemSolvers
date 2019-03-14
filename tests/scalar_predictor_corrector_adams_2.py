import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import RungeKuttaFourthSolver, BackwardEulerSolver
from src.multi_step_solvers import AdamsBashforthTwoSolver, AdamsMoultonOneSolver

from src.predictor_corrector_solvers import PredictorCorrectorSolver

h = lambda u, t: np.array([-2*u[0] + math.cos(t)])
true_value = lambda t: math.exp(-2*t) + (1/5.0)*(2*math.cos(t) + math.sin(t))

de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.2
t_n = 12




problem = IVP(de, u_0, t_0)

first_step_explicit_slv = RungeKuttaFourthSolver(problem, t_n, step)
first_step_implicit_slv = BackwardEulerSolver(problem, t_n, step)

pred_slv = AdamsBashforthTwoSolver(problem, first_step_explicit_slv, t_n, step)

predictor_working = AdamsBashforthTwoSolver(problem, first_step_explicit_slv, t_n, step)
predictor_working.solve()

corr_slv = AdamsMoultonOneSolver(problem, first_step_implicit_slv, t_n, step)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv, adaptive=True)
pred_corr_slv.solve()
pred_corr_slv.print_solution()



comparison = ResultsComparator([pred_corr_slv, predictor_working], true_solution=true_value)
comparison.print_result_graphs()


