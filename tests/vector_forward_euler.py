import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.solver import ForwardEulerSolver


def f(u, t):
    result = np.zeros(len(u))
    result[0] = u[0] + 2*u[1]
    result[1] = 3*u[0] + 2*u[1]
    return result


de = ODE(f)

u_0 = np.array([6, 4])
t_0 = 0

step = 0.01
t_n = 2


problem = IVP(de, u_0, t_0)

slv = ForwardEulerSolver(problem, t_n, step)

slv.solve()

print(slv.solution.value_mesh)




def true_value(t):
    return np.array(
        [4*math.exp(4 * t) + 2*math.exp(-1 * t),
         6*math.exp(-2 * t) + 2*math.exp(-1 * t)]
    )


def print_true_solution(h, start, end):
    t_i = start
    while t_i <= end:
        print("u(", t_i, ") = ", true_value(t_i))
        t_i += h

    print()


print_true_solution(0.2, t_0, t_n)
