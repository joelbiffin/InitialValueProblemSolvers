import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import ForwardEulerSolver


g = lambda u, t: np.array([u[0] * t])

de = ODE(g)

u_0 = np.array([1])
t_0 = 0

step = 0.01
precision = 2
t_n = 5


problem = IVP(de, u_0, t_0)

slv = ForwardEulerSolver(problem, t_n, step, precision)
slv.solve()



def true_value(t):
    return math.exp(0.5 * t ** 2)


def print_true_solution(h, start, end):
    t_i = start
    while t_i <= end:
        print("u(", t_i, ") = ", true_value(t_i))
        t_i += h

    print()


#print_true_solution(step, t_0, t_n)


comparison = ResultsComparator(slv.solution, true_value)

comparison.pointwise_plot(0)


