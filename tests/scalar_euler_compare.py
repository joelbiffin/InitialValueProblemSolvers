import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver


g = lambda u, t: np.array([t**2 - 4*t - 1])

de = ODE(g)

u_0 = np.array([0])
t_0 = 0

step = 0.0001
t_n = 10


problem = IVP(de, u_0, t_0)

forward_slv = ForwardEulerSolver(problem, t_n, step)
forward_slv.solve()

backward_slv = BackwardEulerSolver(problem, t_n, step)
backward_slv.solve()


#print(slv.solution)




def true_value(t):
    return (1 / 3.0) * (t**3) - 2*(t**2) - t


# TODO: update ResultsComparator class to deal with different input types.
forward_comparison = ResultsComparator(forward_slv.solution, true_value)
forward_comparison.print_result_graphs()


backward_comparison = ResultsComparator(backward_slv.solution,true_value)
backward_comparison.print_result_graphs()

