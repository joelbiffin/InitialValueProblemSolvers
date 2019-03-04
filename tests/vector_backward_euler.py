import math
import numpy as np
import time


from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import BackwardEulerSolver


def f(u, t):
    result = np.zeros(len(u))
    result[0] = 8*u[0] - 3*u[1] + math.exp(t)
    result[1] = -2*u[0] + 7*u[1] + math.exp(t)
    return result


g = lambda u, t: np.array([
    8*u[0]     - 3*u[1]    + math.exp(t),
    -2*u[0]    + 7*u[1]    + math.exp(t)
])


de = ODE(g)

u_0 = np.array([4, 2])
t_0 = 0

step = 0.0001
precision = 4
t_n = 2


problem = IVP(de, u_0, t_0)

slv = BackwardEulerSolver(problem, t_n, step, precision)

start_time = time.time()

slv.solve()

end_time = time.time()

print("\nOverall time taken:\t", end_time - start_time)

print(slv.solution.value_mesh)





def true_value(t):
    return np.array(
        [3*math.exp(5*t) + 1.5*math.exp(10*t),
         3*math.exp(5*t) - math.exp(10*t)]
    )


def print_true_solution(h, start, end):
    t_i = start
    counter = 1
    while t_i <= end:
        if counter % 100 == 0:
            print(t_i, ":\t", true_value(t_i))
        t_i += h
        counter += 1

    print()


print_true_solution(step, t_0, t_n)


comparison = ResultsComparator(slv.solution, true_value)
comparison.print_result_graphs()


