import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.predictor_corrector_solvers import PredictorCorrectorSolver
from src.results import ResultsComparator
from src.one_step_solvers import RungeKuttaFourthSolver, BackwardEulerSolver
from src.multi_step_solvers import AdamsMoultonOneSolver, AdamsBashforthTwoSolver

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
t_n = 6



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
