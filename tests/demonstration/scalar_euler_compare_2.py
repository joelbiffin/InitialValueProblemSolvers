import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver

g = lambda u, t: np.array([-100*u[0]])
true_value = lambda t: np.array([math.exp(-100*t)])

de = ODE(g)

u_0 = np.array([1])
t_0 = 0

t_n = 0.2


problem = IVP(de, u_0, t_0)

def compare_methods(step):
    global forward_slv, backward_slv

    forward_slv = ForwardEulerSolver(problem, t_n, step)
    forward_slv.solve()
    print("{}.solve() ran in {}s.".format(str(forward_slv), forward_slv.solve_time))

    backward_slv = BackwardEulerSolver(problem, t_n, step)
    backward_slv.solve()
    print("{}.solve() ran in {}s.".format(str(backward_slv), backward_slv.solve_time))


for h in [0.1, 0.001]:
    compare_methods(h)
    comparison = ResultsComparator([forward_slv, backward_slv], true_solution=true_value)
    comparison.print_result_graphs()
    comparison.compute_global_truncation_errors()
    comparison.graph_global_truncation_errors()
    t_n = 1
