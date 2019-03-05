import math
import numpy as np

from src.ivp import IVP
from src.ode import ODE
from src.results import ResultsComparator
from src.solver import AdamsBashforthSecondSolver, ForwardEulerSolver, RungeKuttaFourthSolver

f = lambda u, t: np.array([
    -10*u[0] + u[1],
    -1*u[1]
])


de = ODE(f)

u_0 = np.array([
    1,
    1
])
t_0 = 0

step = 0.00001
precision = 5
t_n = 5


problem = IVP(de, u_0, t_0)

first_step_slv = RungeKuttaFourthSolver(problem, t_n, step, precision)
adams_slv = AdamsBashforthSecondSolver(problem, first_step_slv, t_n, step, precision)

adams_slv.solve()

print(adams_slv.solution)





def true_value(t):
    return np.array([
        (1 / 9.0) * math.exp(-1*t) + (8 / 9.0) * math.exp(-10 * t),
        math.exp(-1 * t)
    ])


# adams_slv.print_solution()

forward_comparison = ResultsComparator(adams_slv.solution, true_value)
forward_comparison.pointwise_plot(0)


