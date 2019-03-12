import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver
from src.predictor_corrector_solvers import PredictorCorrectorSolver

g = lambda u, t: np.array([math.cos(t)])

de = ODE(g)

u_0 = np.array([0])
t_0 = 0

step = math.pi / 32
t_n = math.pi * 6


problem = IVP(de, u_0, t_0)

pred_slv = ForwardEulerSolver(problem, t_n, step)
corr_slv = BackwardEulerSolver(problem, t_n, step)

pred_corr_slv = PredictorCorrectorSolver(pred_slv, corr_slv)
pred_corr_slv.solve()



def true_value(t):
    return math.sin(t)


pred_corr_slv.print_solution()



comparison = ResultsComparator(pred_corr_slv.solution, true_value)
comparison.print_result_graphs()


