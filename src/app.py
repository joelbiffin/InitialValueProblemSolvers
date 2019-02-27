import numpy as np
import math


from src.ivp import IVP
from src.ode import ODE
from src.solver import ForwardEulerSolver

""""""""""""""""""""""""""""""""""""""""""""""""""""""


def f(u, t):
    return np.array([u * t])


de = ODE(f)

u_0 = np.array([1])
t_0 = 0

step = 0.5
t_n = 5



def true_value(t):
    return math.exp(0.5 * t ** 2)


def print_true_solution(h, start, end):
    t_i = start
    while t_i <= end:
        print("u(", t_i, ") = ", true_value(t_i))
        t_i += h

    print()


print_true_solution(step, t_0, t_n)

input("Done with true solution? \n")

problem = IVP(de, u_0, t_0)

slv = ForwardEulerSolver(problem, t_n, step)

slv.solve()

print(slv.solution)

"""

def f(u, t):
    return np.array([0*u[0] + u[1],
                    -6*u[0] + 5*u[1]])


ode_system = ODE(f)

u_0 = np.array([1, 2])
t_0 = 0

step = 0.5
t_n = 5


system_problem = IVP(f, u_0, t_0)
system_solver = ForwardEulerSolver(system_problem, t_n, step)


#system_solver.ivp.ode.compute_derivative(u_0, t_0)

system_solver.solve()


"""
