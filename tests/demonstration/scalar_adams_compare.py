import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.one_step_solvers import RungeKuttaFourthSolver
from src.multi_step_solvers import *

g = lambda u, t: np.array([math.cos(t)])
true_value = lambda t: np.array([math.sin(t)])

de = ODE(g)

u_0 = np.array([0])
t_0 = 0

t_n = 12


problem = IVP(de, u_0, t_0)

def compare_2_methods(step):
    global ab_2, am_2

    fwd = RungeKuttaFourthSolver(problem, t_n, step)

    ab_2 = AdamsBashforthTwoSolver(problem, fwd, t_n, step)
    ab_2.solve()
    print("{}.solve() ran in {}s.".format(str(ab_2), ab_2.solve_time))

    am_2 = AdamsMoultonTwoSolver(problem, fwd, t_n, step)
    am_2.solve()
    print("{}.solve() ran in {}s.".format(str(am_2), am_2.solve_time))


def compare_3_methods(step):
    global ab_3, am_3

    fwd = RungeKuttaFourthSolver(problem, t_n, step)

    ab_3 = AdamsBashforthThreeSolver(problem, fwd, t_n, step)
    ab_3.solve()
    print("{}.solve() ran in {}s.".format(str(ab_3), ab_3.solve_time))

    am_3 = AdamsMoultonThreeSolver(problem, fwd, t_n, step)
    am_3.solve()
    print("{}.solve() ran in {}s.".format(str(am_3), am_3.solve_time))




for h in [0.1, 0.4]:
    compare_2_methods(h)
    comparison = ResultsComparator([ab_2, am_2], true_solution=true_value)
    comparison.print_result_graphs()
    comparison.compute_global_truncation_errors()
    comparison.graph_global_truncation_errors()

