import math as m
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import ForwardEulerSolver, RungeKuttaFourthSolver

true_value = lambda t: np.array([m.exp(-t)*(3*m.cos(t) + m.sin(t)) \
                          + 6*m.sin(2*t) - 3*m.cos(2*t), 0])

g = lambda u, t: np.array([
    u[1],
    30*m.cos(2*t) - 2*u[0] - 2*u[1]
])

de = ODE(g)


u_0 = np.array([0, 10])
t_0 = 0

t_n = 10


problem = IVP(de, u_0, t_0)

def compare_methods(step):
    global forward_slv, backward_slv

    forward_slv = ForwardEulerSolver(problem, t_n, step / 4.0)
    forward_slv.solve()
    print("{}.solve() ran in {}s.".format(str(forward_slv), forward_slv.solve_time))

    backward_slv = RungeKuttaFourthSolver(problem, t_n, step)
    backward_slv.solve()
    print("{}.solve() ran in {}s.".format(str(backward_slv), backward_slv.solve_time))


for h in [0.1]:
    compare_methods(h)
    comparison = ResultsComparator([forward_slv], true_solution=true_value)
    comparison.print_result_graphs()
    comparison.compute_global_truncation_errors()
    comparison.graph_global_truncation_errors()
    comparison = ResultsComparator([backward_slv], true_solution=true_value)
    comparison.print_result_graphs()
    comparison.compute_global_truncation_errors()
    comparison.graph_global_truncation_errors()

