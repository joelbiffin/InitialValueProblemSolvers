import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import ForwardEulerSolver

h = lambda u, t: np.array([-2*u[0] + math.cos(t)])
true_value = lambda t: math.exp(-2*t) + (1/5.0)*(2*math.cos(t) + math.sin(t))

g = lambda u, t: np.array([-1*u[0]])

de = ODE(h)

u_0 = np.array([1.4])
t_0 = 0

step = 0.5
precision = 3
t_n = 12


problem = IVP(de, u_0, t_0)

slv = ForwardEulerSolver(problem, t_n, step, precision)
slv.solve()



# def true_value(t):
#     return math.exp(-1*t)


def print_true_solution(h, start, end):
    t_i = start
    while t_i <= end:
        print("u(", t_i, ") = ", true_value(t_i))
        t_i += h

    print()


#print_true_solution(step, t_0, t_n)


comparison = ResultsComparator(slv.solution, true_value)
comparison.print_result_graphs()

comparison.compute_local_truncation_errors()
print(comparison.local_truncation_error)


